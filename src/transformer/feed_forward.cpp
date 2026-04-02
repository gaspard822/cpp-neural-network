#include <iostream>
#include <fstream>
#include "transformer/feed_forward.hpp"
#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

FeedForward::FeedForward(ActivationFunction* activation, int seq, int d_model, int d_ff) :
                        activation(activation), seq(seq), d_model(d_model), d_ff(d_ff),
                        X(mx::zeros({1, 1}, mx::float32)), U(mx::zeros({1, 1}, mx::float32)), H(mx::zeros({1, 1}, mx::float32)),
                        W1(mx::zeros({d_model, d_ff}, mx::float32)), W2(mx::zeros({d_ff, d_model}, mx::float32)),
                        d_W1(mx::zeros({d_model, d_ff}, mx::float32)), d_W2(mx::zeros({d_ff, d_model}, mx::float32)),
                        b1(mx::zeros({1, d_ff}, mx::float32)), b2(mx::zeros({1, d_model}, mx::float32)),
                        d_b1(mx::zeros({1, d_ff}, mx::float32)), d_b2(mx::zeros({1, d_model}, mx::float32)) {
    
    if (activation->get_type() == ActivationType::RELU) {
        // He initialization for the weights if using a ReLU activation function
        // This is not the true He initialization as the weights are chosen from a uniform distribution and not a
        // Gaussian one, but it works well in practice and is efficient
        float he_factor_1 = sqrt(2.0f / d_model);
        float he_factor_2 = sqrt(2.0f / d_ff);
        W1 = mx::random::uniform(-he_factor_1, he_factor_1, {d_model, d_ff});
        W2 = mx::random::uniform(-he_factor_2, he_factor_2, {d_ff, d_model});
    } else if (activation->get_type() == ActivationType::SIGMOID) {
        // Glorot initialization for the weights if using a sigmoid activation function
        double glorot_factor = sqrt(6.0f / (d_ff + d_model));
        W1 = mx::random::uniform(-glorot_factor, glorot_factor, {d_model, d_ff});
        W2 = mx::random::uniform(-glorot_factor, glorot_factor, {d_ff, d_model});
    } else {
        W1 = mx::random::uniform(-1.0f, 1.0f, {d_model, d_ff});
        W2 = mx::random::uniform(-1.0f, 1.0f, {d_ff, d_model});
    }

    params = {TrainableParameter(W1, d_W1), TrainableParameter(b1, d_b1), TrainableParameter(W2, d_W2), TrainableParameter(b2, d_b2)};
}

void FeedForward::forward(const mx::array& input) {
    // input : (num_tokens, d_model)
    X = input;
    U = mx::matmul(input, W1) + b1;
    H = activation->apply(U);
    output = mx::matmul(H, W2) + b2;
}

void FeedForward::backward(const mx::array& d_output) {
    d_W2 = d_W2 + mx::matmul(mx::transpose(H), d_output);
    d_b2 = d_b2 + mx::sum(d_output, 0, true);
    mx::array d_U = mx::matmul(d_output, mx::transpose(W2)) * activation->derivative(U);
    d_W1 = d_W1 + mx::matmul(mx::transpose(X), d_U);
    d_b1 = d_b1 + mx::sum(d_U, 0, true);
    d_input = mx::matmul(d_U, mx::transpose(W1));
}

mx::array FeedForward::infer(const mx::array& input) const {
    mx::array U_tmp = mx::matmul(input, W1) + b1;
    mx::array H_tmp = activation->apply(U_tmp);
    return mx::matmul(H_tmp, W2) + b2;
}

const vector<TrainableParameter>& FeedForward::get_parameters() const {
    return params;
}

const mx::array& FeedForward::get_output() const {
    return output;
}

const mx::array& FeedForward::get_d_input() const {
    return d_input;
}

string FeedForward::get_layer_name() const {
    return "FeedForward";
}

string FeedForward::get_activation_name() const {
    return "";
}

LayerType FeedForward::get_type() const {
    return LayerType::FEED_FORWARD;
}

void FeedForward::save(ofstream& file) const {
    file << get_layer_name() << "\n";
    file << seq << " " << d_model << " " << d_ff << "\n";
    save_array(file, W1);
    save_array(file, b1);
    save_array(file, W2);
    save_array(file, b2);
}

void FeedForward::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model >> d_ff;
    W1 = load_array(file);
    b1 = load_array(file);
    W2 = load_array(file);
    b2 = load_array(file);
}
