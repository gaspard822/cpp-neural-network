#include <iostream>
#include "multi_head_attention_layer.hpp"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int seq, int d_model, int h, bool masked) : seq(seq), d_model(d_model), h(h), masked(masked) {
    momentum = 0.9;
    // for now, let d_k = d_v = d_model / h
    d_k = d_model / h;
    d_v = d_model / h;

    gamma = VectorXd::Ones(seq);
    beta = VectorXd::Zero(seq);
    mean = VectorXd::Zero(seq);
    inv_sqrt_var_plus_epsilon = VectorXd::Zero(seq);
    running_mean = VectorXd::Zero(seq);
    running_variance = VectorXd::Zero(seq);

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

// READ IT AGAIN CAREFULLY AND I'M QUITE CONFIDENT THAT THIS FUNCTION IS CORRECT
void MultiHeadAttentionLayer::forward(const MatrixXd& layer_input) {
    // layer_input : (seq, d_model)

    // cout << "MultiHeadAttentionLayer::backward()\n" << endl;
    double epsilon = 1e-8;
    input = layer_input;
    // cout << "input: \n" << input << endl;
    // compute mu and sigma along rows
    mean = input.rowwise().mean();
    // cout << "mean: \n" << mean << endl;

    MatrixXd diff = input.colwise() - mean;
    // cout << "diff: \n" << diff << endl;
    
    VectorXd variance = diff.array().square().rowwise().mean();
    // cout << "variance: \n" << variance << endl;

    inv_sqrt_var_plus_epsilon = VectorXd::Ones(variance.rows()).array() / (variance.array() + epsilon).sqrt();
    // cout << "inv_sqrt_var_plus_epsilon: \n" << inv_sqrt_var_plus_epsilon << endl;

    // NEED TO UPDATE THE RUNNING MEAN AND THE RUNNING VARIANCE

    E_hat = diff.array().colwise() / inv_sqrt_var_plus_epsilon.array();
    // cout << "E_hat: \n" << E_hat << endl;

    E_bar = (E_hat.array().colwise() * gamma.array()).colwise() + beta.array();
    // cout << "E_bar: \n" << E_bar << endl;
    
    for (int i = 0; i < h; i++) {
        Q[i] = E_bar * WQ[i];
        K[i] = E_bar * WK[i];
        V[i] = E_bar * WV[i];
    }
    // cout << "Q[0]: \n" << Q[0] << endl;
    // cout << "K[0]: \n" << K[0] << endl;

    for (int i = 0; i < h; i++) {
        // Don't need to store J_i
        softmaxJ[i] = (Q[i] * K[i].transpose()).array() * (1/sqrt(d_k));
        if (masked) {
            MatrixXd M = MatrixXd::Constant(seq, seq, -__DBL_MAX__);
            M.triangularView<Lower>().setZero();
            softmaxJ[i] += M;
        }
        VectorXd J_i_max = softmaxJ[i].rowwise().maxCoeff();
        softmaxJ[i] = softmaxJ[i] - J_i_max.replicate(1, seq);
        softmaxJ[i] = softmaxJ[i].array().exp();
        VectorXd shifted_softmaxJi_exp_sum = softmaxJ[i].rowwise().sum();
        softmaxJ[i] = softmaxJ[i].array().colwise() / shifted_softmaxJi_exp_sum.array();
        // cout << "softmaxJ[i]: \n" << softmaxJ[i] << endl;
        // cout << "J_i_max: \n" << J_i_max << endl;
        // cout << "shifted_softmaxJi: \n" << shifted_softmaxJi << endl;
        // cout << "shifted_softmaxJi_exp: \n" << shifted_softmaxJi_exp << endl;
        // cout << "shifted_softmaxJi_exp_sum: \n" << shifted_softmaxJi_exp_sum << endl;
        // cout << "softmaxJ[i]: \n" << softmaxJ[i] << endl;
        // cout << "=====================================" << endl;
    }
    // cout << "softmaxJ[1]: \n" << softmaxJ[1] << endl;
    
    // WOULD MAKE SENSE TO DELETE THIS input, output FROM LAYER, AS IT STORES SOME EXTRA MATRICES FOR NO REASON
    // => WOULD NEED TO MODIFY neural_network::forward()

    output = MatrixXd::Zero(seq, d_model);
    // cout << "COMPUTING output" << endl;
    for (int i = 0; i < h; i++) {
        // cout << "(i * d_v, 0, d_v, d_model): \n" << "(" << i * d_v << ", "  << 0 << ", " << d_v << ", " << d_model << ")" << endl;
        head[i] = softmaxJ[i] * V[i];
        // cout << "head[i]: \n" << head[i] << endl;
        output += head[i] * WO[i];
        // cout << "(softmaxJ[i] * V[i]) * WO[i]: \n" << (softmaxJ[i] * V[i]) * WO[i] << endl;
        // cout << "output: \n" << output << endl;
        // cout << "=====================================" << endl;
    }
    output = output + input;
    
}


MatrixXd MultiHeadAttentionLayer::backward(const MatrixXd& d_output) {
    // d_output : (seq, d_model)

    // cout << "MultiHeadAttentionLayer::backward()\n" << endl;
    vector<MatrixXd> d_head;
    for (int i = 0; i < h; i++) {
        d_head.push_back(d_output * WO[i].transpose());
        d_WO[i] = head[i].transpose() * d_output;
    }
    // Could merge the previous loop with this one for efficiency (will do it after it works)
    vector<MatrixXd> d_softmaxJ, d_J, d_Q, d_K, d_V;
    MatrixXd d_E_bar = MatrixXd::Zero(seq, d_model);
    for (int i = 0; i < h; i++) {
        // cout << "=====================================" << endl;
        d_softmaxJ.push_back(d_head[i] * V[i].transpose());
        d_J.push_back(((d_softmaxJ[i].array() - (d_softmaxJ[i].array() * softmaxJ[i].array()).rowwise().sum().replicate(1, seq)).array()).array() * softmaxJ[i].array());
        if (masked) {
            MatrixXd M = MatrixXd::Zero(seq, seq);
            M.triangularView<Lower>().setOnes();
            d_J[i].array() *= M.array();
        }
        // cout << "d_softmaxJ[i]: \n" << d_softmaxJ[i] << endl;
        // cout << "softmaxJ[i]: \n" << softmaxJ[i] << endl;
        // cout << "d_J[i]: \n" << d_J[i] << endl;

        d_V.push_back(softmaxJ[i].transpose() * d_head[i]);
        d_K.push_back((d_J[i].transpose() * Q[i]).array() * (1/sqrt(d_k)));
        d_Q.push_back((d_J[i] * K[i]).array() * (1/sqrt(d_k)));
        d_E_bar += d_Q[i] * WQ[i].transpose() + d_K[i] * WK[i].transpose() + d_V[i] * WV[i].transpose();
        d_WQ[i] = E_bar.transpose() * d_Q[i];
        d_WK[i] = E_bar.transpose() * d_K[i];
        d_WV[i] = E_bar.transpose() * d_V[i];
    }
    d_gamma = (d_E_bar.array() * E_hat.array()).rowwise().sum();
    d_beta = d_E_bar.rowwise().sum();
    // cout << "d_E_bar: \n" << d_E_bar << endl;
    // cout << "d_beta: \n" << d_beta << endl;
    MatrixXd d_E_hat = d_E_bar.array().colwise() * gamma.array();

    // cout << "E'': \n" << input.colwise() - mean << endl;
    // cout << "d_E_hat: \n" << d_E_hat << endl;
    // cout << "inv_sqrt_var: \n" << inv_sqrt_var_plus_epsilon << endl;
    // cout << "d_output: \n" << d_output << endl;
    // cout << "d_model: \n" << d_model << endl;

    // Manually chekced d_E and seemed correct
    MatrixXd d_E = (input.colwise() - mean).array().colwise() * inv_sqrt_var_plus_epsilon.array().pow(3);
    // cout << "\nStep 1: \n" << d_E << endl;
    d_E = d_E.array().colwise() * (d_E_hat.array() * (input.colwise() - mean).array()).rowwise().sum();
    // cout << "\nStep 2: \n" << d_E << endl;
    d_E = d_E.array().colwise() + inv_sqrt_var_plus_epsilon.array() * d_beta.array(); // d_beta is already the sum of d_E_hat along the rows
    // cout << "\nStep 3: \n" << d_E << endl;
    d_E = d_E.array() * (-1.0/d_model) + d_E_hat.array() + d_output.array();
    // cout << "\nStep 4: \n" << d_E << endl;
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