#include <iostream>
#include <fstream>
#include "transformer/layer_norm.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

LayerNorm::LayerNorm(int seq, int d_model) : seq(seq), d_model(d_model), epsilon(1e-8f),
                                             inv_sqrt_var_plus_epsilon(mx::zeros({seq}, mx::float32)),
                                             gamma(mx::ones({1, d_model}, mx::float32)),
                                             beta(mx::zeros({1, d_model}, mx::float32)),
                                             d_gamma(mx::zeros({1, d_model}, mx::float32)),
                                             d_beta(mx::zeros({1, d_model}, mx::float32)),
                                             diff(mx::zeros({1, 1, 1}, mx::float32)),
                                             normalized_input(mx::zeros({1, 1, 1}, mx::float32)) {

    params = {TrainableParameter(gamma, d_gamma), TrainableParameter(beta, d_beta)};
}

void LayerNorm::forward(const mx::array& input) {
    // input: (num_sentences, max_sentence_size, d_model)
    diff = input - mx::mean(input, 2, true);  // per-row average, keep the same dimensions
    mx::array variance = mx::mean(mx::square(diff), 2, false);
    inv_sqrt_var_plus_epsilon = mx::ones({variance.shape(0), variance.shape(1)}, mx::float32) / mx::sqrt(variance + epsilon);
    normalized_input = diff * mx::reshape(inv_sqrt_var_plus_epsilon, {variance.shape(0), variance.shape(1), 1});
    output = normalized_input * gamma + beta;
}

void LayerNorm::backward(const mlx::core::array& d_output) {
    int num_sentences = d_output.shape(0);
    int max_sentence_size = d_output.shape(1);
    d_gamma = mx::sum(mx::sum(d_output * normalized_input, 0), 0);
    d_beta = mx::sum(mx::sum(d_output, 0), 0);
    mx::array d_normalized_input = d_output * gamma;
    mx::array isv = mx::reshape(inv_sqrt_var_plus_epsilon, {num_sentences, max_sentence_size, 1});
    mx::array dot_dn_norm = mx::sum(d_normalized_input * normalized_input, 2, true);
    mx::array sum_dn = mx::sum(d_normalized_input, 2, true);
    d_input = isv * (d_normalized_input - (1.0f / d_model) * (normalized_input * dot_dn_norm + sum_dn));
}

mx::array LayerNorm::infer(const mx::array& input) const {
    // input: (num_tokens, d_model)
    return input;
}

const vector<TrainableParameter>& LayerNorm::get_parameters() const {
    return params;
}

const mx::array& LayerNorm::get_output() const {
    return output;
}

const mx::array& LayerNorm::get_d_input() const {
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

void LayerNorm::save(ofstream& file) const {
    file << get_layer_name() << "\n";
    file << seq << " " << d_model << "\n";
    save_array(file, gamma);
    save_array(file, beta);
}

void LayerNorm::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model;
    gamma = load_array(file);
    beta = load_array(file);
}
