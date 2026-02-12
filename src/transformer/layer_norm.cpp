#include <iostream>
#include "transformer/layer_norm.hpp"


LayerNorm::LayerNorm(int seq, int d_model) : seq(seq), d_model(d_model), epsilon(1e-8) {
    gamma = RowVectorXd::Ones(d_model);
    beta = RowVectorXd::Zero(d_model);
    d_gamma = RowVectorXd(d_model);
    d_beta = RowVectorXd(d_model);
    mean = VectorXd::Zero(seq);
    inv_sqrt_var_plus_epsilon = VectorXd::Zero(seq);
    params = {TrainableParameter(gamma, d_gamma), TrainableParameter(beta, d_beta)};
}

void LayerNorm::forward(const MatrixXd& input) {
    // input: (num_tokens, d_model)
    mean = input.rowwise().mean();  // per-token average
    diff = input.colwise() - mean;  // centered tokens
    VectorXd variance = diff.array().square().rowwise().mean();
    inv_sqrt_var_plus_epsilon = VectorXd::Ones(variance.rows()).array() / (variance.array() + epsilon).sqrt();
    normalized_input = diff.array().colwise() * inv_sqrt_var_plus_epsilon.array();  // normalized tokens
    output = (normalized_input.array().rowwise() * gamma.array()).rowwise() + beta.array();  // normalized tokens scaled w.r.t. the d_model dimension
}

void LayerNorm::backward(const MatrixXd& d_output) {
    d_gamma += (d_output.array() * normalized_input.array()).colwise().sum().matrix();
    d_beta += d_output.colwise().sum();
    MatrixXd d_normalized_input = d_output.array().rowwise() * gamma.array();

    d_input = diff.array().colwise() * inv_sqrt_var_plus_epsilon.array().pow(3);
    d_input = d_input.array().colwise() * (d_normalized_input.array() * diff.array()).rowwise().sum();
    d_input = d_input.array().colwise() + (inv_sqrt_var_plus_epsilon.array() * d_normalized_input.rowwise().sum().array());
    d_input = d_input.array() * (-1.0/d_model) + d_normalized_input.array().colwise() * inv_sqrt_var_plus_epsilon.array();
}

MatrixXd LayerNorm::infer(const MatrixXd& input) const {
    return MatrixXd();
}

const vector<TrainableParameter>& LayerNorm::get_parameters() const {
    return params;
}

const MatrixXd& LayerNorm::get_output() const {
    return output;
}

const MatrixXd& LayerNorm::get_d_input() const {
    return d_input;
}

string LayerNorm::get_layer_name() const {
    return "LayerNorm";
}

string LayerNorm::get_activation_name() const {
    return "";
}

LayerType LayerNorm::get_type() const {
    return LayerType::LAYER_NORM;
}
