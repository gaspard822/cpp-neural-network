#include <iostream>
#include <fstream>
#include "transformer/layer_norm.hpp"
#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

LayerNorm::LayerNorm(int seq, int d_model) : seq(seq), d_model(d_model), epsilon(1e-8),
                                             inv_sqrt_var_plus_epsilon_mlx(mx::zeros({seq}, mx::float32)),
                                             gamma_mlx(mx::ones({1, d_model}, mx::float32)),
                                             beta_mlx(mx::zeros({1, d_model}, mx::float32)),
                                             d_gamma_mlx(mx::zeros({1, d_model}, mx::float32)),
                                             d_beta_mlx(mx::zeros({1, d_model}, mx::float32)),
                                             diff_mlx(mx::zeros({1, 1}, mx::float32)),
                                             normalized_input_mlx(mx::zeros({1, 1}, mx::float32)) {
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

void LayerNorm::forward_mlx(const mx::array& input) {
    // input: (num_tokens, d_model)
    diff_mlx = input - mx::mean(input, 1, true);  // per-row average, keep the same dimensions
    mx::array variance = mx::mean(mx::square(diff_mlx), 1, false);
    inv_sqrt_var_plus_epsilon_mlx = mx::ones({variance.shape(0)}, mx::float32) / mx::sqrt(variance + epsilon);
    normalized_input_mlx = diff_mlx * mx::reshape(inv_sqrt_var_plus_epsilon_mlx, {-1, 1});
    output_mlx = normalized_input_mlx * gamma_mlx + beta_mlx;
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

void LayerNorm::backward_mlx(const mlx::core::array& d_output) {
    d_gamma_mlx = d_gamma_mlx + mx::sum(d_output * normalized_input_mlx, 0);
    d_beta_mlx = d_beta_mlx + mx::sum(d_output, 0);
    mx::array d_normalized_input_mlx = d_output * gamma_mlx;

    mx::array isv = mx::reshape(inv_sqrt_var_plus_epsilon_mlx, {-1, 1});
    mx::array dot_dn_norm = mx::sum(d_normalized_input_mlx * normalized_input_mlx, 1, true);
    mx::array sum_dn = mx::sum(d_normalized_input_mlx, 1, true);
    d_input_mlx = isv * (d_normalized_input_mlx - (1.0f / d_model) * (normalized_input_mlx * dot_dn_norm + sum_dn));
}

MatrixXd LayerNorm::infer(const MatrixXd& input) const {
    // input: (num_tokens, d_model)
    VectorXd mean_tmp = input.rowwise().mean();  // per-token average
    MatrixXd diff_tmp = input.colwise() - mean_tmp;  // centered tokens
    VectorXd variance_tmp = diff_tmp.array().square().rowwise().mean();
    VectorXd inv_sqrt_var_plus_epsilon_tmp = VectorXd::Ones(variance_tmp.rows()).array() / (variance_tmp.array() + epsilon).sqrt();
    MatrixXd normalized_input = diff_tmp.array().colwise() * inv_sqrt_var_plus_epsilon_tmp.array();  // normalized tokens
    return (normalized_input.array().rowwise() * gamma.array()).rowwise() + beta.array();  // normalized tokens scaled w.r.t. the d_model dimension
}

mx::array LayerNorm::infer_mlx(const mx::array& input) const {
    // input: (num_tokens, d_model)
    mx::array diff_tmp_mlx = input - mx::mean(input, 1, true);  // per-row average, keep the same dimensions
    mx::array variance_tmp_mlx = mx::mean(mx::square(diff_tmp_mlx), 1, false);
    mx::array inv_sqrt_var_plus_epsilon_tmp_mlx = mx::ones({variance_tmp_mlx.shape(0)}, mx::float32) / mx::sqrt(variance_tmp_mlx + epsilon);
    mx::array normalized_input_tmp_mlx = diff_tmp_mlx * mx::reshape(inv_sqrt_var_plus_epsilon_tmp_mlx, {-1, 1});
    return normalized_input_tmp_mlx * gamma_mlx + beta_mlx;
}

const vector<TrainableParameter>& LayerNorm::get_parameters() const {
    return params;
}

const MatrixXd& LayerNorm::get_output() const {
    return output;
}

const mx::array& LayerNorm::get_output_mlx() const {
    return output_mlx;
}

const MatrixXd& LayerNorm::get_d_input() const {
    return d_input;
}

const mx::array& LayerNorm::get_d_input_mlx() const {
    return d_input_mlx;
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

void LayerNorm::save(ofstream& file) const {
    file << get_layer_name() << "\n";
    file << seq << " " << d_model << "\n";
    file << gamma << "\n";
    file << beta << "\n";
}

void LayerNorm::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model;
    for (int i = 0; i < d_model; i++) {
        file >> gamma(i);
    }
    for (int i = 0; i < d_model; i++) {
        file >> beta(i);
    }
}
