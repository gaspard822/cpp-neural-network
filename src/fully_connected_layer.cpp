#include <iostream>
#include <chrono>
#include "fully_connected_layer.hpp"
#include "identity.hpp"
#include "relu.hpp"

using namespace std;

FCGradients::FCGradients(MatrixXd& dw, VectorXd& dbi, RowVectorXd& dg, RowVectorXd& dbe):
    d_weights(dw), d_bias(dbi), d_gamma(dg), d_beta(dbe) {}

FullyConnectedLayer::FullyConnectedLayer(ActivationFunction* activation,
                                         int input_size, int output_size) : activation(activation) {
    momentum = 0.9;

    if (activation->get_type() == ActivationType::RELU) {
        // He initialization for the weights if using a ReLU activation function
        // This is not the true He initialization as the weights are chosen from a uniform distribution and not a
        // Gaussian one, but it works well in practice and is efficient
        weights = MatrixXd::Random(output_size, input_size) * sqrt(2.0 / input_size);
    } else if (activation->get_type() == ActivationType::SIGMOID) {
        // Glorot initialization for the weights if using a sigmoid activation function
        double limit = sqrt(6.0 / (input_size + output_size));
        weights = MatrixXd::Random(output_size, input_size) * limit;
    } else {
        weights = MatrixXd::Random(output_size, input_size);
    }

    bias = VectorXd::Zero(output_size);
    gamma = RowVectorXd::Ones(input_size);
    beta = RowVectorXd::Zero(input_size);
    running_mean = RowVectorXd::Zero(input_size);
    running_variance = RowVectorXd::Zero(input_size);
    inv_sqrt_var_plus_epsilon = RowVectorXd::Zero(input_size);
    d_weights = MatrixXd(output_size, input_size);
    d_bias = VectorXd(output_size);
    d_gamma = RowVectorXd(input_size);
    d_beta = RowVectorXd(input_size);

}

FullyConnectedLayer::FullyConnectedLayer(ActivationFunction* activation, 
                                         const MatrixXd& init_weights,
                                         const VectorXd& init_bias,
                                         const RowVectorXd& init_gamma,
                                         const RowVectorXd& init_beta) : activation(activation) {
    momentum = 0.9;
    weights = init_weights;
    bias = init_bias;
    gamma = init_gamma;
    beta = init_beta;
    running_mean = RowVectorXd::Zero(init_weights.cols());
    running_variance = RowVectorXd::Zero(init_weights.cols());
    inv_sqrt_var_plus_epsilon = RowVectorXd::Zero(init_weights.cols());
    int input_size = init_weights.cols();
    int output_size = init_weights.rows();
    d_weights = MatrixXd(output_size, input_size);
    d_bias = VectorXd(output_size);
    d_gamma = RowVectorXd(input_size);
    d_beta = RowVectorXd(input_size);

}

FullyConnectedLayer::~FullyConnectedLayer() {
    delete activation;
}

void FullyConnectedLayer::forward(const MatrixXd& layer_input) {
    double epsilon = 1e-8;
    input = layer_input;
    // Compute the feature-wise mean
    RowVectorXd mean = input.colwise().mean();
    // Compute the normalized input and store it in a_hat
    MatrixXd diff = input.rowwise() - mean;
    RowVectorXd variance = (diff.array().square().colwise().sum()) / input.rows();
    inv_sqrt_var_plus_epsilon = RowVectorXd::Ones(variance.cols()).array() / (variance.array().sqrt() + epsilon);
    a_hat = diff.array().rowwise() * (inv_sqrt_var_plus_epsilon.array());
    // Scale and shift the normalized input
    a_bar = (a_hat.array().rowwise() * gamma.array()).rowwise() + beta.array();
    // Multiply by the weights and add the bias
    z = (a_bar * weights.transpose()).rowwise() + bias.transpose();
    // Apply the activation function
    output = activation->apply(z);

    // Update the running mean and running variance
    running_mean = momentum * running_mean + (1-momentum) * mean;
    running_variance = momentum * running_variance + (1-momentum) * variance;
}

MatrixXd FullyConnectedLayer::backward(const MatrixXd& d_output) {
    int num_points = d_output.rows();
    // Compute the derivative w.r.t. to z the unactivated output
    MatrixXd dz;
    if (activation->get_type() == ActivationType::IDENTITY) {
        dz = d_output;
    } else {
        dz = d_output.cwiseProduct(activation->derivative(z));
    }
    // Compute the derivative w.r.t. to the weights and the bias
    d_weights = (dz.transpose() * a_bar) / num_points;
    d_bias = (dz.colwise().sum()) / num_points;
    // Compute the derivative w.r.t. to the normalized, scaled and shifted input
    MatrixXd da_bar = dz * weights;
    // Compute the derivative w.r.t. to the scale and the shift
    d_gamma = (da_bar.cwiseProduct(a_hat)).colwise().sum() / num_points;
    d_beta = (da_bar.colwise().sum()) / num_points;
    // Compute the derivative w.r.t. to the normalized input
    MatrixXd da_hat = da_bar.array().rowwise() * gamma.array();
    // Compute the derivative w.r.t. to the input and return it
    MatrixXd da = da_hat.array().rowwise() * (inv_sqrt_var_plus_epsilon.array());
    return da;
}

MatrixXd FullyConnectedLayer::infer(const MatrixXd& layer_input) const {
    double epsilon = 1e-8;
    // Normalize the input w.r.t. to the running mean and running variance
    MatrixXd diff = layer_input.rowwise() - running_mean;
    RowVectorXd running_inv_sqrt_var_plus_epsilon = RowVectorXd::Ones(running_variance.cols()).array() / (running_variance.array().sqrt() + epsilon);
    MatrixXd input_hat = diff.array().rowwise() * (running_inv_sqrt_var_plus_epsilon.array());
    // Scale and shift the normalized input
    MatrixXd input_bar = (input_hat.array().rowwise() * gamma.array()).rowwise() + beta.array();
    // Multiply by the weights, add the bias
    MatrixXd z_input = (input_bar * weights.transpose()).rowwise() + bias.transpose();
    // Apply the activation function and return
    return activation->apply(z_input);
}

unique_ptr<Gradients> FullyConnectedLayer::get_gradients() {
    return make_unique<FCGradients>(d_weights, d_bias, d_gamma, d_beta);
}

unique_ptr<Gradients> FullyConnectedLayer::get_params() {
    return make_unique<FCGradients>(weights, bias, gamma, beta);
}

const MatrixXd& FullyConnectedLayer::get_weights() const {
    return weights;
}

const VectorXd& FullyConnectedLayer::get_bias() const {
    return bias;
}

const MatrixXd& Layer::get_output() const {
    return output;
}

const RowVectorXd& FullyConnectedLayer::get_gamma() const {
    return gamma;
}

const RowVectorXd& FullyConnectedLayer::get_beta() const {
    return beta;
}

const RowVectorXd& FullyConnectedLayer::get_running_mean() const {
    return running_mean;
}

const RowVectorXd& FullyConnectedLayer::get_running_variance() const {
    return running_variance;
}

const RowVectorXd& FullyConnectedLayer::get_inv_sqrt_var_plus_epsilon() const {
    return inv_sqrt_var_plus_epsilon;
}

void FullyConnectedLayer::set_running_mean(RowVectorXd new_running_mean) {
    running_mean = new_running_mean;
}

void FullyConnectedLayer::set_running_variance(RowVectorXd new_running_variance) {
    running_variance = new_running_variance;
}

void FullyConnectedLayer::set_inv_sqrt_var_plus_epsilon(RowVectorXd new_inv_sqrt_var_plus_epsilon) {
    inv_sqrt_var_plus_epsilon = new_inv_sqrt_var_plus_epsilon;
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