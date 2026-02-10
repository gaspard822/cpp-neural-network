#include <iostream>
#include "transformer/multi_head_attention.hpp"

MultiHeadAttention::MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode) :
        seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), mode(mode) {
    
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
        WQ[i] = MatrixXd::Random(d_model, d_k) * limit_d_k;
        WK[i] = MatrixXd::Random(d_model, d_k) * limit_d_k;
        WV[i] = MatrixXd::Random(d_model, d_v) * limit_d_v;
        WO[i] = MatrixXd::Random(d_v, d_model) * limit_d_v;
    }

    Q.resize(h);
    K.resize(h);
    V.resize(h);

    softmaxJ.resize(h);
    head.resize(h);
}

void MultiHeadAttention::forward(const MatrixXd& input) {
    // input : (num_tokens, d_model)
    cout << "========== MultiHeadAttention::forward() ==========" << endl;  // debug
    int num_tokens;
    if (!is_cross_attention()) {
        num_tokens = input.rows();
    } else {
        // if we are doing cross-attention, the number of tokens is the max between the input of this layer and the encoder output
        num_tokens = input.rows() > encoder_output.rows() ? input.rows() : encoder_output.rows();
    }
    X = input;
    for (int i = 0; i < h; i++) {
        Q[i] = input * WQ[i];
        if (!is_cross_attention()) {
            K[i] = input * WK[i];
            V[i] = input * WV[i];
        } else {
            if (encoder_output.size() == 0) throw runtime_error("encoder_output not set for cross-attention");
            K[i] = encoder_output * WK[i];
            V[i] = encoder_output * WV[i];
            cout << "encoder_output (" << encoder_output.rows() << "," << encoder_output.cols() << "):" << endl << encoder_output << endl; // debug
            // TODO: NEED TO ADD PADDING TO MAKE SURE THAT Q[i], WHICH COMES FROM THE DECODER, HAS THE SAME DIMENSIONS AS K[i] AND V[i]
        }
        softmaxJ[i] = (Q[i] * K[i].transpose()).array() * (1/sqrt(d_k));
        if (is_masked_attention()) {
            MatrixXd M = MatrixXd::Constant(num_tokens, num_tokens, -1e15);
            M.triangularView<Lower>().setZero();
            softmaxJ[i] += M;
        }
        cout << "J[i] (" << softmaxJ[i].rows() << "," << softmaxJ[i].cols() << "):" << endl << softmaxJ[i] << endl; // debug
        VectorXd J_i_max = softmaxJ[i].rowwise().maxCoeff();
        cout << "J_i_max:" << endl << J_i_max << endl; // debug
        softmaxJ[i] = softmaxJ[i] - J_i_max.replicate(1, num_tokens);
        softmaxJ[i] = softmaxJ[i].array().exp();
        VectorXd shifted_softmaxJi_exp_sum = softmaxJ[i].rowwise().sum();
        softmaxJ[i] = softmaxJ[i].array().colwise() / shifted_softmaxJi_exp_sum.array();
        cout << "softmaxJ[i] (" << softmaxJ[i].rows() << "," << softmaxJ[i].cols() << "):" << endl << softmaxJ[i] << endl; // debug
    }

    output = MatrixXd::Zero(num_tokens, d_model);
    for (int i = 0; i < h; i++) {
        head[i] = softmaxJ[i] * V[i];
        output += head[i] * WO[i];
    }

    output = output + input;
    cout << "+++ output (" << output.rows() << "," << output.cols() << "):" << endl << output << endl << endl; // debug
}

void MultiHeadAttention::backward(const MatrixXd& d_output) {
    // d_output : (seq, d_model)
    d_input = d_output;
    if (is_cross_attention()) d_encoder_output = MatrixXd::Zero(seq, d_model);
    for (int i = 0; i < h; i++) {
        MatrixXd d_head = d_output * WO[i].transpose();
        d_WO[i] = head[i].transpose() * d_output;
        MatrixXd d_softmaxJ = d_head * V[i].transpose();
        VectorXd row_dot = (d_softmaxJ.array() * softmaxJ[i].array()).rowwise().sum();
        MatrixXd d_J = softmaxJ[i].array() * (d_softmaxJ.array().colwise() - row_dot.array());
        
        if (is_masked_attention()) {
            MatrixXd M = MatrixXd::Zero(seq, seq);
            M.triangularView<Lower>().setOnes();
            d_J.array() *= M.array();
        }

        MatrixXd d_V = softmaxJ[i].transpose() * d_head;
        MatrixXd d_K = (d_J.transpose() * Q[i]).array() * (1/sqrt(d_k));
        MatrixXd d_Q = (d_J * K[i]).array() * (1/sqrt(d_k));
        d_WQ[i] = X.transpose() * d_Q;
        if (!is_cross_attention()) {
            d_WK[i] = X.transpose() * d_K;
            d_WV[i] = X.transpose() * d_V;
            d_input += d_Q * WQ[i].transpose() + d_K * WK[i].transpose() + d_V * WV[i].transpose();
        } else {
            d_WK[i] = encoder_output.transpose() * d_K;
            d_WV[i] = encoder_output.transpose() * d_V;
            d_input += d_Q * WQ[i].transpose();
            d_encoder_output += d_K * WK[i].transpose() + d_V * WV[i].transpose();
        }
    }
}

MatrixXd MultiHeadAttention::infer(const MatrixXd& input) const {
    return MatrixXd();
}

void MultiHeadAttention::set_encoder_output(const MatrixXd& enc_out) {
    encoder_output = enc_out;
}

unique_ptr<Gradients> MultiHeadAttention::get_gradients() {
    return nullptr;
}

unique_ptr<Gradients> MultiHeadAttention::get_params() {
    return nullptr;
}

const MatrixXd& MultiHeadAttention::get_output() const {
    return output;
}

const MatrixXd& MultiHeadAttention::get_d_input() const {
    return d_input;
}

const MatrixXd& MultiHeadAttention::get_d_encoder_output() const {
    if (!is_cross_attention()) throw runtime_error("get_d_encoder_output() can only be called for cross-attention");
    return d_encoder_output;
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
