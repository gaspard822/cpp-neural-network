#include "transformer/layer_norm.hpp"

/*
LayerNorm::LayerNorm(int seq, int d_model) : epsilon(1e-8) {
    gamma = RowVectorXd::Ones(d_model);
    beta = RowVectorXd::Zero(d_model);
    mean = VectorXd::Zero(seq);
    inv_sqrt_var_plus_epsilon = VectorXd::Zero(seq);
}

void LayerNorm::forward(const MatrixXd& input) {
    // layer_input : (seq, d_model)
    // FIXME: Make sure that input refers to the passed arguments and not to the variable of the class Layer
    mean = input.rowwise().mean();  // per-token average
    MatrixXd diff = input.colwise() - mean;  // centered tokens
    VectorXd variance = diff.array().square().rowwise().mean();
    inv_sqrt_var_plus_epsilon = VectorXd::Ones(variance.rows()).array() / (variance.array() + epsilon).sqrt();

    input_normed = diff.array().colwise() / inv_sqrt_var_plus_epsilon.array();  // normalized tokens
    input_normed_and_scaled = (input_normed.array().rowwise() * gamma.array()).rowwise() + beta.array();  // normalized tokens scaled w.r.t. the d_model dimension
    // return input_normed_and_scaled;
}
*/
/*
MatrixXd LayerNorm::backward(const MatrixXd& d_output) {
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
*/