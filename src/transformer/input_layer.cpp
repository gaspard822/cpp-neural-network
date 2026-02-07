#include <iostream>
#include <fstream>
#include <array>
#include "transformer/input_layer.hpp"

InputLayer::InputLayer(int seq, int d_model) : seq(seq), d_model(d_model) {
    int PAD_ID = 0;
    // Initialize the embeddings matrix
    double limit = sqrt(6.0 / (vocab_size + d_model));
    embeddings = MatrixXd::Random(vocab_size, d_model) * limit;
    embeddings.row(PAD_ID).setZero();
}

void InputLayer::forward(const MatrixXd& layer_input) {
    // layer_input : (seq, d_model)
    
}

void InputLayer::backward(const MatrixXd& d_output) {
    // d_output : (seq, d_model)
    
}

MatrixXd InputLayer::infer(const MatrixXd& layer_input) const {
    return MatrixXd();
}

unique_ptr<Gradients> InputLayer::get_gradients() {
    return nullptr;
}

unique_ptr<Gradients> InputLayer::get_params() {
    return nullptr;
}

const MatrixXd& InputLayer::get_output() const {
    return output;
}

const MatrixXd& InputLayer::get_d_input() const {
    return d_input;
}

string InputLayer::get_activation_name() const {
    return "";
}

LayerType InputLayer::get_type() const {
    return LayerType::INPUT_LAYER;
}
