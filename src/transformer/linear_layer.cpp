#include <iostream>
#include <fstream>
#include "transformer/linear_layer.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

LinearLayer::LinearLayer(int d_model, int vocab_size) : d_model(d_model), vocab_size(vocab_size),
                                                        X(mx::zeros({1, 1}, mx::float32)),
                                                        W(mx::zeros({d_model, vocab_size}, mx::float32)), d_W(mx::zeros({d_model, vocab_size}, mx::float32)),
                                                        b(mx::zeros({1, vocab_size}, mx::float32)), d_b(mx::zeros({1, vocab_size}, mx::float32)) {
    
    // Glorot initialization
    float glorot_factor = sqrt(6.0f / (d_model + vocab_size));
    W = mx::random::uniform(-glorot_factor, glorot_factor, {d_model, vocab_size});

    params = {TrainableParameter(W, d_W), TrainableParameter(b, d_b)};
}

void LinearLayer::forward(const mx::array& input) {
    // input : (num_tokens, d_model)
    X = input;
    output = mx::matmul(input, W) + b;
}

void LinearLayer::backward(const mlx::core::array& d_output) {
    d_W = mx::sum(mx::matmul(mx::transpose(X, {0, 2, 1}), d_output), 0);
    d_b = mx::sum(mx::sum(d_output, 0), 0, true);
    d_input = mx::matmul(d_output, mx::transpose(W));
}

mx::array LinearLayer::infer(const mx::array& input) const {
    return mx::matmul(input, W) + b;
}

const vector<TrainableParameter>& LinearLayer::get_parameters() const {
    return params;
}

const mx::array& LinearLayer::get_output() const {
    return output;
}

const mx::array& LinearLayer::get_d_input() const {
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

void LinearLayer::save(ofstream& file) const {
    file << get_layer_name() << "\n";
    file << d_model << " " << vocab_size << "\n";
    save_array(file, W);
    save_array(file, b);
}

void LinearLayer::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> d_model >> vocab_size;
    W = load_array(file);
    b = load_array(file);
}
