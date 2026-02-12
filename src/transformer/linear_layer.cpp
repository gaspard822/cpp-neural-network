#include <iostream>
#include "transformer/linear_layer.hpp"

LinearLayer::LinearLayer(int d_model, int vocab_size) : d_model(d_model), vocab_size(vocab_size) {
    
    W = MatrixXd::Random(d_model, vocab_size);
    // Glorot initialization
    double limit = sqrt(6.0 / (d_model + vocab_size));
    W = W * limit;

    b = RowVectorXd::Zero(vocab_size);
    
    d_W = MatrixXd(d_model, vocab_size);
    d_b = RowVectorXd(vocab_size);

    params = {TrainableParameter(W, d_W), TrainableParameter(b, d_b)};
}

void LinearLayer::forward(const MatrixXd& input) {
    // input : (num_tokens, d_model)
    X = input;
    output = (input * W).rowwise() + b;
}

void LinearLayer::backward(const MatrixXd& d_output) {
    d_W += X.transpose() * d_output;
    d_b += d_output.colwise().sum();
    d_input = d_output * W.transpose();
}

MatrixXd LinearLayer::infer(const MatrixXd& input) const {
    return MatrixXd();
}

const vector<TrainableParameter>& LinearLayer::get_parameters() const {
    return params;
}

const MatrixXd& LinearLayer::get_output() const {
    return output;
}

const MatrixXd& LinearLayer::get_d_input() const {
    return d_input;
}

string LinearLayer::get_layer_name() const {
    return "LinearLayer";
}

string LinearLayer::get_activation_name() const {
    return "";
}

LayerType LinearLayer::get_type() const {
    return LayerType::LINEAR_LAYER;
}
