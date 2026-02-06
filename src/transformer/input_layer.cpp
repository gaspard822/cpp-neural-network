#include <iostream>
#include <fstream>
#include <array>
#include "transformer/input_layer.hpp"

InputLayer::InputLayer(int seq, int d_model, const string& path) : seq(seq), d_model(d_model), path(path) {
    // Count how many unique character tokens we have
    array<bool, 256> tokens{};
    tokens.fill(false);
    ifstream input_file(path, ios::binary);
    if (!input_file) throw runtime_error("Cannot open file: " + path);
    unsigned char c;
    while (input_file.read(reinterpret_cast<char*>(&c), 1)) {
        tokens[c] = true;
    }
    int unique_chars = 0;
    for (bool b : tokens) if (b) ++unique_chars;
    vocab_size = unique_chars + 3;  // we will have three special tokens: <SOS>, <EOS> and <PAD>
    
    // Initialize the embeddings matrix
    double limit = sqrt(6.0 / (vocab_size + d_model));
    embeddings = MatrixXd::Random(vocab_size, d_model) * limit;
    embeddings.row(PAD_ID).setZero();
}

void InputLayer::forward(const MatrixXd& layer_input) {
    // layer_input : (seq, d_model)
    
}

MatrixXd InputLayer::backward(const MatrixXd& d_output) {
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
string InputLayer::get_activation_name() const {
    return "";
}
LayerType InputLayer::get_type() const {
    return LayerType::INPUT_LAYER;
}
