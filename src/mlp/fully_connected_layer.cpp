#include <iostream>
#include "mlp/fully_connected_layer.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

FullyConnectedLayer::FullyConnectedLayer(ActivationFunction* activation,
                                         int input_size, int output_size) : activation(activation),
                                         weights(mx::zeros({output_size, input_size}, mx::float32)), bias(mx::zeros({output_size}, mx::float32)),
                                         gamma(mx::ones({1, input_size}, mx::float32)), beta(mx::zeros({1, input_size}, mx::float32)),
                                         running_mean(mx::zeros({1, input_size}, mx::float32)), running_variance(mx::zeros({1, input_size}, mx::float32)),
                                         inv_sqrt_var_plus_epsilon(mx::zeros({1, input_size}, mx::float32)),
                                         d_weights(mx::zeros({output_size, input_size}, mx::float32)), d_bias(mx::zeros({output_size}, mx::float32)),
                                         d_gamma(mx::zeros({1, input_size}, mx::float32)), d_beta(mx::zeros({1, input_size}, mx::float32)),
                                         a_hat(mx::zeros({1}, mx::float32)), a_bar(mx::zeros({1}, mx::float32)), z(mx::zeros({1}, mx::float32)) {
    momentum = 0.9f;

    if (activation->get_type() == ActivationType::RELU) {
        // He initialization for the weights if using a ReLU activation function
        float he_factor = sqrt(2.0f / input_size);
        weights = mx::random::uniform(-he_factor, he_factor, {output_size, input_size});
    } else if (activation->get_type() == ActivationType::SIGMOID) {
        // Glorot initialization for the weights if using a sigmoid activation function
        float glorot_factor = sqrt(6.0f / (input_size + output_size));
        weights = mx::random::uniform(-glorot_factor, glorot_factor, {output_size, input_size});
    } else {
        weights = mx::random::uniform(-1.0f, 1.0f, {output_size, input_size});
    }

    params = {TrainableParameter(weights, d_weights), TrainableParameter(bias, d_bias), TrainableParameter(gamma, d_gamma), TrainableParameter(beta, d_beta)};
}

FullyConnectedLayer::~FullyConnectedLayer() {
    delete activation;
}

void FullyConnectedLayer::forward(const mx::array& input) {
    float epsilon = 1e-8f;
    int N = input.shape(0);

    mx::array mean = mx::mean(input, 0, true);
    mx::array diff = input - mean;
    mx::array variance = mx::sum(mx::square(diff), 0, true) / N;
    inv_sqrt_var_plus_epsilon = mx::array(1.0f) / mx::sqrt(variance + epsilon);
    a_hat = diff * inv_sqrt_var_plus_epsilon;
    a_bar = a_hat * gamma + beta;
    z = mx::matmul(a_bar, mx::transpose(weights)) + bias;
    output = activation->apply(z);

    running_mean = momentum * running_mean + (1.0 - momentum) * mean;
    running_variance = momentum * running_variance + (1.0 - momentum) * variance;
}

void FullyConnectedLayer::backward(const mx::array& d_output) {
    int N = d_output.shape(0);

    mx::array dz = d_output * activation->derivative(z);

    d_weights = mx::matmul(mx::transpose(dz), a_bar) / N;
    d_bias = mx::sum(dz, 0) / N;

    mx::array da_bar = mx::matmul(dz, weights);
    d_gamma = mx::sum(da_bar * a_hat, 0, true) / N;
    d_beta = mx::sum(da_bar, 0, true) / N;

    mx::array da_hat = da_bar * gamma;
    d_input = da_hat * inv_sqrt_var_plus_epsilon;
}

mx::array FullyConnectedLayer::infer(const mx::array& layer_input) const {
    float epsilon = 1e-8f;
    mx::array diff = layer_input - running_mean;
    mx::array running_inv = mx::array(1.0f) / mx::sqrt(running_variance + epsilon);
    mx::array input_hat = diff * running_inv;
    mx::array input_bar = input_hat * gamma + beta;
    mx::array z_input = mx::matmul(input_bar, mx::transpose(weights)) + bias;
    return activation->apply(z_input);
}

const vector<TrainableParameter>& FullyConnectedLayer::get_parameters() const {
    return params;
}

const mx::array& FullyConnectedLayer::get_output() const {
    return output;
}

const mx::array& FullyConnectedLayer::get_d_input() const {
    return d_input;
}

string FullyConnectedLayer::get_layer_name() const {
    return "FullyConnectedLayer";
}

string FullyConnectedLayer::get_activation_name() const {
    if (activation->get_type() == ActivationType::RELU) {
        return "relu";
    } else if (activation->get_type() == ActivationType::SIGMOID) {
        return "sigmoid";
    } else if (activation->get_type() == ActivationType::IDENTITY) {
        return "identity";
    }
    else {
        return "activation not recognized";
    }
}

LayerType FullyConnectedLayer::get_type() const {
    return LayerType::FULLY_CONNECTED_LAYER;
}

void FullyConnectedLayer::save(ofstream& file) const {
    file << "FullyConnectedLayer\n";
    file << get_activation_name() << "\n";
    file << weights.shape(1) << " " << weights.shape(0) << "\n"; // input_size output_size
    save_array(file, weights);
    save_array(file, bias);
    save_array(file, gamma);
    save_array(file, beta);
    save_array(file, running_mean);
    save_array(file, running_variance);
}

void FullyConnectedLayer::load(ifstream& file) {
    weights = load_array(file);
    bias = load_array(file);
    gamma = load_array(file);
    beta = load_array(file);
    running_mean = load_array(file);
    running_variance = load_array(file);
}
