#include <iostream>
#include "multi_head_attention_layer.hpp"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int seq, int d_model, int h, AttentionMode mode, MultiHeadAttentionLayer* encoder_mha) :
        seq(seq), d_model(d_model), h(h), mode(mode), encoder_mha(encoder_mha) {
    
    if (mode == AttentionMode::DECODER_CROSS && !encoder_mha) throw runtime_error("The corresponding encoder was not provided");
    if (mode == AttentionMode::DECODER_CROSS && encoder_mha->get_mode() != AttentionMode::ENCODER_SELF) throw runtime_error("The given mha is not an encoder");

    momentum = 0.9;
    // for now, let d_k = d_v = d_model / h
    d_k = d_model / h;
    d_v = d_model / h;

    gamma_self = RowVectorXd::Ones(d_model);
    beta_self = RowVectorXd::Zero(d_model);
    if (is_cross_attention()) {
        gamma_cross = RowVectorXd::Ones(d_model);
        beta_cross = RowVectorXd::Zero(d_model);
    }
    mean = VectorXd::Zero(seq);
    inv_sqrt_var_plus_epsilon = VectorXd::Zero(seq);

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

void MultiHeadAttentionLayer::forward(const MatrixXd& layer_input) {
    // layer_input : (seq, d_model)
    double epsilon = 1e-8;
    input = layer_input;
    mean = input.rowwise().mean();

    MatrixXd diff = input.colwise() - mean;
    VectorXd variance = diff.array().square().rowwise().mean();
    inv_sqrt_var_plus_epsilon = VectorXd::Ones(variance.rows()).array() / (variance.array() + epsilon).sqrt();

    E_hat = diff.array().colwise() / inv_sqrt_var_plus_epsilon.array();
    E_bar = (E_hat.array().rowwise() * gamma_self.array()).rowwise() + beta_self.array();

    for (int i = 0; i < h; i++) {
        Q[i] = E_bar * WQ[i];
        K[i] = E_bar * WK[i];
        V[i] = E_bar * WV[i];
        softmaxJ[i] = (Q[i] * K[i].transpose()).array() * (1/sqrt(d_k));
        if (is_masked_attention()) {
            MatrixXd M = MatrixXd::Constant(seq, seq, -1e15);
            M.triangularView<Lower>().setZero();
            softmaxJ[i] += M;
        }
        VectorXd J_i_max = softmaxJ[i].rowwise().maxCoeff();
        softmaxJ[i] = softmaxJ[i] - J_i_max.replicate(1, seq);
        softmaxJ[i] = softmaxJ[i].array().exp();
        VectorXd shifted_softmaxJi_exp_sum = softmaxJ[i].rowwise().sum();
        softmaxJ[i] = softmaxJ[i].array().colwise() / shifted_softmaxJi_exp_sum.array();
    }
    
    // TODO: WOULD MAKE SENSE TO DELETE THIS input, output FROM LAYER, AS IT STORES SOME EXTRA MATRICES FOR NO REASON
    // => WOULD NEED TO MODIFY neural_network::forward()

    output = MatrixXd::Zero(seq, d_model);
    for (int i = 0; i < h; i++) {
        head[i] = softmaxJ[i] * V[i];
        output += head[i] * WO[i];
    }

    output = output + input;
}


MatrixXd MultiHeadAttentionLayer::backward(const MatrixXd& d_output) {
    // d_output : (seq, d_model)
    cout << "MultiHeadAttentionLayer::backward()\n" << endl;
    MatrixXd d_E_bar = MatrixXd::Zero(seq, d_model);
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
        d_E_bar += d_Q * WQ[i].transpose() + d_K * WK[i].transpose() + d_V * WV[i].transpose();
        d_WQ[i] = E_bar.transpose() * d_Q;
        d_WK[i] = E_bar.transpose() * d_K;
        d_WV[i] = E_bar.transpose() * d_V;
    }

    d_gamma_self = (d_E_bar.array() * E_hat.array()).colwise().sum();
    d_beta_self = d_E_bar.colwise().sum();
    MatrixXd d_E_hat = d_E_bar.array().rowwise() * gamma_self.array();

    MatrixXd diff = input.colwise() - mean;
    MatrixXd d_E = diff.array().colwise() * inv_sqrt_var_plus_epsilon.array().pow(3);
    d_E = d_E.array().colwise() * (d_E_hat.array() * diff.array()).rowwise().sum();
    d_E = d_E.array().colwise() + (inv_sqrt_var_plus_epsilon.array() * d_E_hat.rowwise().sum().array());
    d_E = d_E.array() * (-1.0/d_model) + d_E_hat.array().colwise() * inv_sqrt_var_plus_epsilon.array() + d_output.array();
    return d_E;
}

MatrixXd MultiHeadAttentionLayer::infer(const MatrixXd& layer_input) const {
    return MatrixXd();
}
unique_ptr<Gradients> MultiHeadAttentionLayer::get_gradients() {
    return nullptr;
}
unique_ptr<Gradients> MultiHeadAttentionLayer::get_params() {
    return nullptr;
}
string MultiHeadAttentionLayer::get_activation_name() const {
    return "";
}
LayerType MultiHeadAttentionLayer::get_type() const {
    return LayerType::MULTI_HEAD_ATTENTION_LAYER;
}

AttentionMode MultiHeadAttentionLayer::get_mode() const {
    return mode;
}

bool MultiHeadAttentionLayer::is_cross_attention() const {
    return mode == AttentionMode::DECODER_CROSS;
}

bool MultiHeadAttentionLayer::is_masked_attention() const {
    return mode == AttentionMode::DECODER_MASKED_SELF;
}