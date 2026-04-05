#include <iostream>
#include <fstream>
#include "transformer/multi_head_attention.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

MultiHeadAttention::MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode) :
        seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), mode(mode),
        X(mx::zeros({1, 1}, mx::float32)), encoder_output(mx::zeros({1, 1}, mx::float32)), d_encoder_output(mx::zeros({1, 1}, mx::float32)),
        WQ(mx::zeros({d_model, h*d_k}, mx::float32)), WK(mx::zeros({d_model, h*d_k}, mx::float32)), WV(mx::zeros({d_model, h*d_v}, mx::float32)), WO(mx::zeros({h*d_v, d_model}, mx::float32)),
        d_WQ(mx::zeros({d_model, h*d_k}, mx::float32)), d_WK(mx::zeros({d_model, h*d_k}, mx::float32)), d_WV(mx::zeros({d_model, h*d_v}, mx::float32)), d_WO(mx::zeros({h*d_v, d_model}, mx::float32)),
        Q(mx::zeros({1, 1}, mx::float32)), K(mx::zeros({1, 1}, mx::float32)), V(mx::zeros({1, 1}, mx::float32)),
        softmaxJ(mx::zeros({1, 1}, mx::float32)), head(mx::zeros({1, 1}, mx::float32)), 
        forward_mask(mx::zeros({1, 1}, mx::float32)), backward_mask(mx::zeros({1, 1}, mx::float32)) {
    
    params = {};
    // Glorot initialization for the parameter matrices
    float glorot_factor_d_k = sqrt(6.0f / (d_model + d_k));
    float glorot_factor_d_v = sqrt(6.0f / (d_model + d_v));
    // Initialize weights
    WQ = mx::random::uniform(-glorot_factor_d_k, glorot_factor_d_k, {d_model, h * d_k});
    WK = mx::random::uniform(-glorot_factor_d_k, glorot_factor_d_k, {d_model, h * d_k});
    WV = mx::random::uniform(-glorot_factor_d_v, glorot_factor_d_v, {d_model, h * d_v});
    WO = mx::random::uniform(-glorot_factor_d_v, glorot_factor_d_v, {h * d_v, d_model});
    
    // Add parameters matrices to params
    params.push_back(TrainableParameter(WQ, d_WQ));
    params.push_back(TrainableParameter(WK, d_WK));
    params.push_back(TrainableParameter(WV, d_WV));
    params.push_back(TrainableParameter(WO, d_WO));

    if (is_masked_attention()) {
        // Forward mask: upper triangle filled with -1e15, lower triangle and diagonal are 0
        mx::array ones_upper = mx::triu(mx::ones({seq, seq}, mx::float32), 1);  // strict upper triangle
        forward_mask = ones_upper * -1e15f;

        // Backward mask: lower triangle and diagonal filled with 1, upper triangle is 0
        backward_mask = mx::tril(mx::ones({seq, seq}, mx::float32), 0);
    }
}

void MultiHeadAttention::forward(const mx::array& input) {
    int num_q_tokens = input.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output.shape(0) : input.shape(0);
    X = input;
    // Compute Q=input*WQ and reshape it into {h, num_q_tokens, d_k}
    Q = mx::transpose(mx::reshape(mx::matmul(input, WQ), {num_q_tokens, h, d_k}), {1, 0, 2});
    const mx::array& kv_input = is_cross_attention() ? encoder_output : input;
    // Compute K=input*WK and reshape it into {h, num_kv_tokens, d_k}
    K = mx::transpose(mx::reshape(mx::matmul(kv_input, WK), {num_kv_tokens, h, d_k}), {1, 0, 2});
    // Compute V=input*WV and reshape it into {h, num_kv_tokens, d_v}
    V = mx::transpose(mx::reshape(mx::matmul(kv_input, WV), {num_kv_tokens, h, d_v}), {1, 0, 2});

    mx::array scores = mx::matmul(Q, mx::transpose(K, {0, 2, 1})) / (sqrt((float)d_k));
    if (is_masked_attention()) {
        scores = scores + mx::reshape(mx::slice(forward_mask, {0, 0}, {num_q_tokens, num_kv_tokens}), {1, num_q_tokens, num_kv_tokens});
    }

    softmaxJ = mx::softmax(scores, -1);  // has shape {h, num_q_tokens, num_kv_tokens}
    head = mx::matmul(softmaxJ, V);  // has shape {h, num_q_tokens, d_v}
    mx::array concat_heads = mx::reshape(mx::transpose(head, {1, 0, 2}), {num_q_tokens, h * d_v});  // has shape {num_q_tokens, h*d_v}
    output = mx::matmul(concat_heads, WO);
}

void MultiHeadAttention::backward(const mx::array& d_output) {
    int num_q_tokens = X.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output.shape(0) : X.shape(0);
    d_input = mx::zeros({num_q_tokens, d_model}, mx::float32);
    if (is_cross_attention()) d_encoder_output = mx::zeros({num_kv_tokens, d_model}, mx::float32);

    mx::array d_head = mx::matmul(d_output, mx::transpose(WO));  // has shape {num_q_tokens, h*d_v}
    d_head = mx::transpose(mx::reshape(d_head, {num_q_tokens, h, d_v}), {1, 0, 2}); // has shape {h, num_q_tokens, d_v}

    mx::array concat_heads = mx::reshape(mx::transpose(head, {0, 2, 1}), {h*d_v, num_q_tokens});  // has shape {h*d_v, num_q_tokens}
    d_WO = d_WO + mx::matmul(concat_heads, d_output);  // has shape {h*d_v, d_model}

    mx::array d_softmaxJ = mx::matmul(d_head, mx::transpose(V, {0, 2, 1}));  // has shape {h, num_q_tokens, num_kv_tokens}
    mx::array row_dot = mx::sum(d_softmaxJ * softmaxJ, -1, true);  // has shape {h, num_q_tokens, num_kv_tokens}
    mx::array d_scores = softmaxJ * (d_softmaxJ - row_dot);  // has shape {h, num_q_tokens, num_kv_tokens}
    if (is_masked_attention()) {
        d_scores = d_scores * mx::reshape(mx::slice(backward_mask, {0, 0}, {num_q_tokens, num_kv_tokens}), {1, num_q_tokens, num_kv_tokens});
    }
    mx::array d_V = mx::matmul(mx::transpose(softmaxJ, {0, 2, 1}), d_head);  // has shape {h, num_kv_tokens, d_v}
    mx::array d_K = mx::matmul(mx::transpose(d_scores, {0, 2, 1}), Q) / (sqrt((float)d_k));  // has shape {h, num_kv_tokens, d_k}
    mx::array d_Q = mx::matmul(d_scores, K) / (sqrt((float)d_k));  // has shape {h, num_q_tokens, d_k}

    mx::array d_Q_flat = mx::reshape(mx::transpose(d_Q, {1, 0, 2}), {num_q_tokens, h * d_k});
    mx::array d_K_flat = mx::reshape(mx::transpose(d_K, {1, 0, 2}), {num_kv_tokens, h * d_k});
    mx::array d_V_flat = mx::reshape(mx::transpose(d_V, {1, 0, 2}), {num_kv_tokens, h * d_v});

    d_WQ = d_WQ + mx::matmul(mx::transpose(X), d_Q_flat);  // has shape {d_model, h*d_k}
    if (!is_cross_attention()) {
        d_WK = d_WK + mx::matmul(mx::transpose(X), d_K_flat);  // has shape {d_model, h*d_k}
        d_WV = d_WV + mx::matmul(mx::transpose(X), d_V_flat);  // has shape {d_model, h*d_v}
        d_input = mx::matmul(d_Q_flat, mx::transpose(WQ)) + mx::matmul(d_K_flat, mx::transpose(WK)) + mx::matmul(d_V_flat, mx::transpose(WV));  // has shape {num_q_tokens, d_model}
    } else {
        d_WK = d_WK + mx::matmul(mx::transpose(encoder_output), d_K_flat);  // has shape {d_model, h*d_k}
        d_WV = d_WV + mx::matmul(mx::transpose(encoder_output), d_V_flat);  // has shape {d_model, h*d_v}
        d_input = mx::matmul(d_Q_flat, mx::transpose(WQ));  // has shape {num_q_tokens, d_model}
        d_encoder_output = mx::matmul(d_K_flat, mx::transpose(WK)) + mx::matmul(d_V_flat, mx::transpose(WV));  // has shape {num_kv_tokens, d_model}
    }
}

mx::array MultiHeadAttention::infer(const mx::array& input) const {
    return input;
}

void MultiHeadAttention::set_encoder_output(const mx::array& enc_out) {
    encoder_output = enc_out;
}

const vector<TrainableParameter>& MultiHeadAttention::get_parameters() const {
    return params;
}

const mx::array& MultiHeadAttention::get_output() const {
    return output;
}

const mx::array& MultiHeadAttention::get_d_input() const {
    return d_input;
}

const mx::array& MultiHeadAttention::get_d_encoder_output() const {
    if (!is_cross_attention()) throw runtime_error("get_d_encoder_output() can only be called for cross-attention");
    return d_encoder_output;
}

string MultiHeadAttention::get_layer_name() const {
    if (mode == AttentionMode::ENCODER_SELF) return "MultiHeadAttentionSelf";
    if (mode == AttentionMode::DECODER_MASKED_SELF) return "MultiHeadAttentionMasked";
    return "MultiHeadAttentionCross";
}

string MultiHeadAttention::get_activation_name() const {
    return "";
}

LayerType MultiHeadAttention::get_type() const {
    return LayerType::MULTI_HEAD_ATTENTION_LAYER;
}

AttentionMode MultiHeadAttention::get_mode() const {
    return mode;
}

bool MultiHeadAttention::is_cross_attention() const {
    return mode == AttentionMode::DECODER_CROSS;
}

bool MultiHeadAttention::is_masked_attention() const {
    return mode == AttentionMode::DECODER_MASKED_SELF;
}

void MultiHeadAttention::save(ofstream& file) const {
    file << get_layer_name() << "\n";
    file << seq << " " << d_model << " " << h << " " << d_k << " " << d_v << "\n";
    save_array(file, WQ);
    save_array(file, WK);
    save_array(file, WV);
    save_array(file, WO);
}

void MultiHeadAttention::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model >> h >> d_k >> d_v;
    WQ = load_array(file);
    WK = load_array(file);
    WV = load_array(file);
    WO = load_array(file);
}
