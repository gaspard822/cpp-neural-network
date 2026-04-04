#include <iostream>
#include <fstream>
#include "transformer/multi_head_attention.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

MultiHeadAttention::MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode) :
        seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), mode(mode),
        X(mx::zeros({1, 1}, mx::float32)), encoder_output(mx::zeros({1, 1}, mx::float32)), d_encoder_output(mx::zeros({1, 1}, mx::float32)),
        forward_mask(mx::zeros({1, 1}, mx::float32)), backward_mask(mx::zeros({1, 1}, mx::float32)) {
    
    params = {};
    // Glorot initialization for the parameter matrices
    float glorot_factor_d_k = sqrt(6.0f / (d_model + d_k));
    float glorot_factor_d_v = sqrt(6.0f / (d_model + d_v));
    for (int i = 0; i < h; i++) {
        // Initialize weights
        WQ.push_back(mx::random::uniform(-glorot_factor_d_k, glorot_factor_d_k, {d_model, d_k}));
        WK.push_back(mx::random::uniform(-glorot_factor_d_k, glorot_factor_d_k, {d_model, d_k}));
        WV.push_back(mx::random::uniform(-glorot_factor_d_v, glorot_factor_d_v, {d_model, d_v}));
        WO.push_back(mx::random::uniform(-glorot_factor_d_v, glorot_factor_d_v, {d_v, d_model}));

        // Initialize gradients
        d_WQ.push_back(mx::zeros({d_model, d_k}, mx::float32));
        d_WK.push_back(mx::zeros({d_model, d_k}, mx::float32));
        d_WV.push_back(mx::zeros({d_model, d_v}, mx::float32));
        d_WO.push_back(mx::zeros({d_v, d_model}, mx::float32));
        
        // Add parameters matrices to params
        params.push_back(TrainableParameter(WQ[i], d_WQ[i]));
        params.push_back(TrainableParameter(WK[i], d_WK[i]));
        params.push_back(TrainableParameter(WV[i], d_WV[i]));
        params.push_back(TrainableParameter(WO[i], d_WO[i]));
    }

    Q.assign(h, mx::zeros({1, 1}, mx::float32));
    K.assign(h, mx::zeros({1, 1}, mx::float32));
    V.assign(h, mx::zeros({1, 1}, mx::float32));
    softmaxJ.assign(h, mx::zeros({1, 1}, mx::float32));
    head.assign(h, mx::zeros({1, 1}, mx::float32));

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
    output = mx::zeros({num_q_tokens, d_model}, mx::float32);
    for (int i = 0; i < h; i++) {
        Q[i] = mx::matmul(input, WQ[i]);
        if (!is_cross_attention()) {
            K[i] = mx::matmul(input, WK[i]);
            V[i] = mx::matmul(input, WV[i]);
        } else {
            K[i] = mx::matmul(encoder_output, WK[i]);
            V[i] = mx::matmul(encoder_output, WV[i]);
        }
        mx::array scores = mx::matmul(Q[i], mx::transpose(K[i])) * (1.0f / sqrt((float)d_k));
        if (is_masked_attention()) {
            scores = scores + mx::slice(forward_mask, {0, 0}, {num_q_tokens, num_kv_tokens});
        }
        softmaxJ[i] = mx::softmax(scores, -1);
        head[i] = mx::matmul(softmaxJ[i], V[i]);
        output = output + mx::matmul(head[i], WO[i]);
    }
}

void MultiHeadAttention::backward(const mx::array& d_output) {
    int num_q_tokens = X.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output.shape(0) : X.shape(0);
    d_input = mx::zeros({num_q_tokens, d_model}, mx::float32);
    if (is_cross_attention()) d_encoder_output = mx::zeros({num_kv_tokens, d_model}, mx::float32);
    for (int i = 0; i < h; i++) {
        mx::array d_head = mx::matmul(d_output, mx::transpose(WO[i]));
        d_WO[i] = d_WO[i] + mx::matmul(mx::transpose(head[i]), d_output);
        mx::array d_softmaxJ = mx::matmul(d_head, mx::transpose(V[i]));
        mx::array row_dot = mx::sum(d_softmaxJ * softmaxJ[i], 1, true);
        mx::array d_J = softmaxJ[i] * (d_softmaxJ - row_dot);

        if (is_masked_attention()) {
            d_J = d_J * mx::slice(backward_mask, {0, 0}, {num_q_tokens, num_kv_tokens});
        }

        mx::array d_V = mx::matmul(mx::transpose(softmaxJ[i]), d_head);
        mx::array d_K = mx::matmul(mx::transpose(d_J), Q[i]) * (1.0f / sqrt((float)d_k));
        mx::array d_Q = mx::matmul(d_J, K[i]) * (1.0f / sqrt((float)d_k));
        d_WQ[i] = d_WQ[i] + mx::matmul(mx::transpose(X), d_Q);
        if (!is_cross_attention()) {
            d_WK[i] = d_WK[i] + mx::matmul(mx::transpose(X), d_K);
            d_WV[i] = d_WV[i] + mx::matmul(mx::transpose(X), d_V);
            d_input = d_input + mx::matmul(d_Q, mx::transpose(WQ[i]))
                        + mx::matmul(d_K, mx::transpose(WK[i]))
                        + mx::matmul(d_V, mx::transpose(WV[i]));
        } else {
            d_WK[i] = d_WK[i] + mx::matmul(mx::transpose(encoder_output), d_K);
            d_WV[i] = d_WV[i] + mx::matmul(mx::transpose(encoder_output), d_V);
            d_input = d_input + mx::matmul(d_Q, mx::transpose(WQ[i]));
            d_encoder_output = d_encoder_output
                                 + mx::matmul(d_K, mx::transpose(WK[i]))
                                 + mx::matmul(d_V, mx::transpose(WV[i]));
        }
    }
}

mx::array MultiHeadAttention::infer(const mx::array& input) const {
    int num_q_tokens = input.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output.shape(0) : input.shape(0);
    mx::array output_tmp = mx::zeros({num_q_tokens, d_model}, mx::float32);
    for (int i = 0; i < h; i++) {
        mx::array Q_tmp = mx::matmul(input, WQ[i]);
        mx::array K_tmp = is_cross_attention() ? mx::matmul(encoder_output, WK[i]) : mx::matmul(input, WK[i]);
        mx::array V_tmp = is_cross_attention() ? mx::matmul(encoder_output, WV[i]) : mx::matmul(input, WV[i]);
        mx::array scores = mx::matmul(Q_tmp, mx::transpose(K_tmp)) * (1.0f / sqrt((float)d_k));
        if (is_masked_attention()) {
            scores = scores + mx::slice(forward_mask, {0, 0}, {num_q_tokens, num_kv_tokens});
        }
        mx::array softmax_tmp = mx::softmax(scores, -1);
        output_tmp = output_tmp + mx::matmul(mx::matmul(softmax_tmp, V_tmp), WO[i]);
    }
    return output_tmp;
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
    for (int i = 0; i < h; i++) {
        save_array(file, WQ[i]);
        save_array(file, WK[i]);
        save_array(file, WV[i]);
        save_array(file, WO[i]);
    }
}

void MultiHeadAttention::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model >> h >> d_k >> d_v;
    for (int i = 0; i < h; i++) {
        WQ[i] = load_array(file);
        WK[i] = load_array(file);
        WV[i] = load_array(file);
        WO[i] = load_array(file);
    }
}
