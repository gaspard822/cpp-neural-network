#include <iostream>
#include <fstream>
#include "transformer/multi_head_attention.hpp"
#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

MultiHeadAttention::MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode) :
        seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), mode(mode),
        X_mlx(mx::zeros({1, 1}, mx::float32)), encoder_output_mlx(mx::zeros({1, 1}, mx::float32)), d_encoder_output_mlx(mx::zeros({1, 1}, mx::float32)),
        forward_mask_mlx(mx::zeros({1, 1}, mx::float32)), backward_mask_mlx(mx::zeros({1, 1}, mx::float32)) {
    
    params = {};
    // Glorot initialization for the parameter matrices
    WQ.resize(h);
    WK.resize(h);
    WV.resize(h);
    WO.resize(h);
    d_WQ.resize(h);
    d_WK.resize(h);
    d_WV.resize(h);
    d_WO.resize(h);
    double limit_d_k = sqrt(6.0 / (d_model + d_k));
    double limit_d_v = sqrt(6.0 / (d_model + d_v));
    for (int i = 0; i < h; i++) {
        // Initialize weights
        WQ[i] = MatrixXd::Random(d_model, d_k) * limit_d_k;
        WK[i] = MatrixXd::Random(d_model, d_k) * limit_d_k;
        WV[i] = MatrixXd::Random(d_model, d_v) * limit_d_v;
        WO[i] = MatrixXd::Random(d_v, d_model) * limit_d_v;

        WQ_mlx.push_back(eigen_to_mlx(WQ[i]));
        WK_mlx.push_back(eigen_to_mlx(WK[i]));
        WV_mlx.push_back(eigen_to_mlx(WV[i]));
        WO_mlx.push_back(eigen_to_mlx(WO[i]));

        // Initialize gradients
        d_WQ[i] = MatrixXd(d_model, d_k);
        d_WK[i] = MatrixXd(d_model, d_k);
        d_WV[i] = MatrixXd(d_model, d_v);
        d_WO[i] = MatrixXd(d_v, d_model);

        d_WQ_mlx.push_back(mx::zeros({d_model, d_k}, mx::float32));
        d_WK_mlx.push_back(mx::zeros({d_model, d_k}, mx::float32));
        d_WV_mlx.push_back(mx::zeros({d_model, d_v}, mx::float32));
        d_WO_mlx.push_back(mx::zeros({d_v, d_model}, mx::float32));
        
        // Add parameters matrices to params
        params.push_back(TrainableParameter(WQ[i], d_WQ[i]));
        params.push_back(TrainableParameter(WK[i], d_WK[i]));
        params.push_back(TrainableParameter(WV[i], d_WV[i]));
        params.push_back(TrainableParameter(WO[i], d_WO[i]));
    }

    Q.resize(h);
    K.resize(h);
    V.resize(h);

    softmaxJ.resize(h);
    head.resize(h);

    if (is_masked_attention()) {
        // Forward mask: upper triangle filled with -1e15, lower triangle is 0
        forward_mask = MatrixXd::Constant(seq, seq, -1e15);
        forward_mask.triangularView<Lower>().setZero();

        // Backward mask: lower triangle filled with 1, upper triangle is 0
        backward_mask = MatrixXd::Zero(seq, seq);
        backward_mask.triangularView<Lower>().setOnes();
    }

    Q_mlx.assign(h, mx::zeros({1, 1}, mx::float32));
    K_mlx.assign(h, mx::zeros({1, 1}, mx::float32));
    V_mlx.assign(h, mx::zeros({1, 1}, mx::float32));
    softmaxJ_mlx.assign(h, mx::zeros({1, 1}, mx::float32));
    head_mlx.assign(h, mx::zeros({1, 1}, mx::float32));

    if (is_masked_attention()) {
        forward_mask_mlx = eigen_to_mlx(forward_mask);
        backward_mask_mlx = eigen_to_mlx(backward_mask);
    }
}

void MultiHeadAttention::forward(const MatrixXd& input) {
    // input : (num_tokens, d_model)
    int num_q_tokens = input.rows();
    int num_kv_tokens;
    if (!is_cross_attention()) {
        num_kv_tokens = input.rows();
    } else {
        num_kv_tokens = encoder_output.rows();
    }
    X = input;
    output = MatrixXd::Zero(num_q_tokens, d_model);
    for (int i = 0; i < h; i++) {
        Q[i] = input * WQ[i];
        if (!is_cross_attention()) {
            K[i] = input * WK[i];
            V[i] = input * WV[i];
        } else {
            if (encoder_output.size() == 0) throw runtime_error("encoder_output not set for cross-attention");
            K[i] = encoder_output * WK[i];
            V[i] = encoder_output * WV[i];
        }
        softmaxJ[i] = (Q[i] * K[i].transpose()).array() * (1/sqrt(d_k));
        if (is_masked_attention()) {
            softmaxJ[i] += forward_mask.topLeftCorner(num_q_tokens, num_kv_tokens);
        }
        VectorXd J_i_max = softmaxJ[i].rowwise().maxCoeff();
        softmaxJ[i] = softmaxJ[i] - J_i_max.replicate(1, num_kv_tokens);
        softmaxJ[i] = softmaxJ[i].array().exp();
        VectorXd shifted_softmaxJi_exp_sum = softmaxJ[i].rowwise().sum();
        softmaxJ[i] = softmaxJ[i].array().colwise() / shifted_softmaxJi_exp_sum.array();

        head[i] = softmaxJ[i] * V[i];
        output += head[i] * WO[i];
    }
}

void MultiHeadAttention::forward_mlx(const mx::array& input) {
    int num_q_tokens = input.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output_mlx.shape(0) : input.shape(0);
    X_mlx = input;
    output_mlx = mx::zeros({num_q_tokens, d_model}, mx::float32);
    for (int i = 0; i < h; i++) {
        Q_mlx[i] = mx::matmul(input, WQ_mlx[i]);
        if (!is_cross_attention()) {
            K_mlx[i] = mx::matmul(input, WK_mlx[i]);
            V_mlx[i] = mx::matmul(input, WV_mlx[i]);
        } else {
            K_mlx[i] = mx::matmul(encoder_output_mlx, WK_mlx[i]);
            V_mlx[i] = mx::matmul(encoder_output_mlx, WV_mlx[i]);
        }
        mx::array scores = mx::matmul(Q_mlx[i], mx::transpose(K_mlx[i])) * (1.0f / sqrt((float)d_k));
        if (is_masked_attention()) {
            scores = scores + mx::slice(forward_mask_mlx, {0, 0}, {num_q_tokens, num_kv_tokens});
        }
        softmaxJ_mlx[i] = mx::softmax(scores, -1);
        head_mlx[i] = mx::matmul(softmaxJ_mlx[i], V_mlx[i]);
        output_mlx = output_mlx + mx::matmul(head_mlx[i], WO_mlx[i]);
    }
}

void MultiHeadAttention::backward(const MatrixXd& d_output) {
    int num_q_tokens = X.rows();
    int num_kv_tokens;
    if (!is_cross_attention()) {
        num_kv_tokens = X.rows();
    } else {
        num_kv_tokens = encoder_output.rows();
    }
    d_input = MatrixXd::Zero(num_q_tokens, d_model);
    if (is_cross_attention()) d_encoder_output = MatrixXd::Zero(num_kv_tokens, d_model);
    for (int i = 0; i < h; i++) {
        MatrixXd d_head = d_output * WO[i].transpose();
        d_WO[i] += head[i].transpose() * d_output;
        MatrixXd d_softmaxJ = d_head * V[i].transpose();
        VectorXd row_dot = (d_softmaxJ.array() * softmaxJ[i].array()).rowwise().sum();
        MatrixXd d_J = softmaxJ[i].array() * (d_softmaxJ.array().colwise() - row_dot.array());
        
        if (is_masked_attention()) {
            d_J.array() *= backward_mask.topLeftCorner(num_q_tokens, num_kv_tokens).array();
        }

        MatrixXd d_V = softmaxJ[i].transpose() * d_head;
        MatrixXd d_K = (d_J.transpose() * Q[i]).array() * (1/sqrt(d_k));
        MatrixXd d_Q = (d_J * K[i]).array() * (1/sqrt(d_k));
        d_WQ[i] += X.transpose() * d_Q;
        if (!is_cross_attention()) {
            d_WK[i] += X.transpose() * d_K;
            d_WV[i] += X.transpose() * d_V;
            d_input += d_Q * WQ[i].transpose() + d_K * WK[i].transpose() + d_V * WV[i].transpose();
        } else {
            d_WK[i] += encoder_output.transpose() * d_K;
            d_WV[i] += encoder_output.transpose() * d_V;
            d_input += d_Q * WQ[i].transpose();
            d_encoder_output += d_K * WK[i].transpose() + d_V * WV[i].transpose();
        }
    }
}

void MultiHeadAttention::backward_mlx(const mx::array& d_output) {
    int num_q_tokens = X_mlx.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output_mlx.shape(0) : X_mlx.shape(0);
    d_input_mlx = mx::zeros({num_q_tokens, d_model}, mx::float32);
    if (is_cross_attention()) d_encoder_output_mlx = mx::zeros({num_kv_tokens, d_model}, mx::float32);
    for (int i = 0; i < h; i++) {
        mx::array d_head = mx::matmul(d_output, mx::transpose(WO_mlx[i]));
        d_WO_mlx[i] = d_WO_mlx[i] + mx::matmul(mx::transpose(head_mlx[i]), d_output);
        mx::array d_softmaxJ = mx::matmul(d_head, mx::transpose(V_mlx[i]));
        mx::array row_dot = mx::sum(d_softmaxJ * softmaxJ_mlx[i], 1, true);
        mx::array d_J = softmaxJ_mlx[i] * (d_softmaxJ - row_dot);

        if (is_masked_attention()) {
            d_J = d_J * mx::slice(backward_mask_mlx, {0, 0}, {num_q_tokens, num_kv_tokens});
        }

        mx::array d_V = mx::matmul(mx::transpose(softmaxJ_mlx[i]), d_head);
        mx::array d_K = mx::matmul(mx::transpose(d_J), Q_mlx[i]) * (1.0f / sqrt((float)d_k));
        mx::array d_Q = mx::matmul(d_J, K_mlx[i]) * (1.0f / sqrt((float)d_k));
        d_WQ_mlx[i] = d_WQ_mlx[i] + mx::matmul(mx::transpose(X_mlx), d_Q);
        if (!is_cross_attention()) {
            d_WK_mlx[i] = d_WK_mlx[i] + mx::matmul(mx::transpose(X_mlx), d_K);
            d_WV_mlx[i] = d_WV_mlx[i] + mx::matmul(mx::transpose(X_mlx), d_V);
            d_input_mlx = d_input_mlx + mx::matmul(d_Q, mx::transpose(WQ_mlx[i]))
                        + mx::matmul(d_K, mx::transpose(WK_mlx[i]))
                        + mx::matmul(d_V, mx::transpose(WV_mlx[i]));
        } else {
            d_WK_mlx[i] = d_WK_mlx[i] + mx::matmul(mx::transpose(encoder_output_mlx), d_K);
            d_WV_mlx[i] = d_WV_mlx[i] + mx::matmul(mx::transpose(encoder_output_mlx), d_V);
            d_input_mlx = d_input_mlx + mx::matmul(d_Q, mx::transpose(WQ_mlx[i]));
            d_encoder_output_mlx = d_encoder_output_mlx
                                 + mx::matmul(d_K, mx::transpose(WK_mlx[i]))
                                 + mx::matmul(d_V, mx::transpose(WV_mlx[i]));
        }
    }
}

MatrixXd MultiHeadAttention::infer(const MatrixXd& input) const {
    // input : (num_tokens, d_model)
    int num_q_tokens = input.rows();
    int num_kv_tokens;
    if (!is_cross_attention()) {
        num_kv_tokens = input.rows();
    } else {
        num_kv_tokens = encoder_output.rows();
    }

    vector<MatrixXd> Q_tmp, K_tmp, V_tmp, softmaxJ_tmp;
    Q_tmp.resize(h);
    K_tmp.resize(h);
    V_tmp.resize(h);
    softmaxJ_tmp.resize(h);
    MatrixXd output_tmp = MatrixXd::Zero(num_q_tokens, d_model);
    for (int i = 0; i < h; i++) {
        Q_tmp[i] = input * WQ[i];
        if (!is_cross_attention()) {
            K_tmp[i] = input * WK[i];
            V_tmp[i] = input * WV[i];
        } else {
            if (encoder_output.size() == 0) throw runtime_error("encoder_output not set for cross-attention");
            K_tmp[i] = encoder_output * WK[i];
            V_tmp[i] = encoder_output * WV[i];
        }
        softmaxJ_tmp[i] = (Q_tmp[i] * K_tmp[i].transpose()).array() * (1/sqrt(d_k));
        if (is_masked_attention()) {
            softmaxJ_tmp[i] += forward_mask.topLeftCorner(num_q_tokens, num_kv_tokens);
        }
        VectorXd J_i_max = softmaxJ_tmp[i].rowwise().maxCoeff();
        softmaxJ_tmp[i] = softmaxJ_tmp[i] - J_i_max.replicate(1, num_kv_tokens);
        softmaxJ_tmp[i] = softmaxJ_tmp[i].array().exp();
        VectorXd shifted_softmaxJi_exp_sum = softmaxJ_tmp[i].rowwise().sum();
        softmaxJ_tmp[i] = softmaxJ_tmp[i].array().colwise() / shifted_softmaxJi_exp_sum.array();

        output_tmp += (softmaxJ_tmp[i] * V_tmp[i]) * WO[i];
    }

    return output_tmp;
}

mx::array MultiHeadAttention::infer_mlx(const mx::array& input) const {
    int num_q_tokens = input.shape(0);
    int num_kv_tokens = is_cross_attention() ? encoder_output_mlx.shape(0) : input.shape(0);
    mx::array output_tmp = mx::zeros({num_q_tokens, d_model}, mx::float32);
    for (int i = 0; i < h; i++) {
        mx::array Q_tmp = mx::matmul(input, WQ_mlx[i]);
        mx::array K_tmp = is_cross_attention() ? mx::matmul(encoder_output_mlx, WK_mlx[i]) : mx::matmul(input, WK_mlx[i]);
        mx::array V_tmp = is_cross_attention() ? mx::matmul(encoder_output_mlx, WV_mlx[i]) : mx::matmul(input, WV_mlx[i]);
        mx::array scores = mx::matmul(Q_tmp, mx::transpose(K_tmp)) * (1.0f / sqrt((float)d_k));
        if (is_masked_attention()) {
            scores = scores + mx::slice(forward_mask_mlx, {0, 0}, {num_q_tokens, num_kv_tokens});
        }
        mx::array softmax_tmp = mx::softmax(scores, -1);
        output_tmp = output_tmp + mx::matmul(mx::matmul(softmax_tmp, V_tmp), WO_mlx[i]);
    }
    return output_tmp;
}

void MultiHeadAttention::set_encoder_output(const MatrixXd& enc_out) {
    encoder_output = enc_out;
}

void MultiHeadAttention::set_encoder_output_mlx(const mx::array& enc_out) {
    encoder_output_mlx = enc_out;
}

const vector<TrainableParameter>& MultiHeadAttention::get_parameters() const {
    return params;
}

const MatrixXd& MultiHeadAttention::get_output() const {
    return output;
}

const mx::array& MultiHeadAttention::get_output_mlx() const {
    return output_mlx;
}

const MatrixXd& MultiHeadAttention::get_d_input() const {
    return d_input;
}

const mx::array& MultiHeadAttention::get_d_input_mlx() const {
    return d_input_mlx;
}

const MatrixXd& MultiHeadAttention::get_d_encoder_output() const {
    if (!is_cross_attention()) throw runtime_error("get_d_encoder_output() can only be called for cross-attention");
    return d_encoder_output;
}

const mx::array& MultiHeadAttention::get_d_encoder_output_mlx() const {
    if (!is_cross_attention()) throw runtime_error("get_d_encoder_output_mlx() can only be called for cross-attention");
    return d_encoder_output_mlx;
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
        file << WQ[i] << "\n";
        file << WK[i] << "\n";
        file << WV[i] << "\n";
        file << WO[i] << "\n";
    }
}

void MultiHeadAttention::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model >> h >> d_k >> d_v;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < d_k; k++) {
                file >> WQ[i](j, k);
            }
        }
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < d_k; k++) {
                file >> WK[i](j, k);
            }
        }
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < d_v; k++) {
                file >> WV[i](j, k);
            }
        }
        for (int j = 0; j < d_v; j++) {
            for (int k = 0; k < d_model; k++) {
                file >> WO[i](j, k);
            }
        }
    }
}