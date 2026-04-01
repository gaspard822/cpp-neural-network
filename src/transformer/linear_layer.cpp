#include <iostream>
#include <fstream>
#include "transformer/linear_layer.hpp"
#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

LinearLayer::LinearLayer(int d_model, int vocab_size) : d_model(d_model), vocab_size(vocab_size),
                                                        X_mlx(mx::zeros({1, 1}, mx::float32)),
                                                        W_mlx(mx::zeros({d_model, vocab_size}, mx::float32)), d_W_mlx(mx::zeros({d_model, vocab_size}, mx::float32)),
                                                        b_mlx(mx::zeros({1, vocab_size}, mx::float32)), d_b_mlx(mx::zeros({1, vocab_size}, mx::float32)) {
    
    W = MatrixXd::Random(d_model, vocab_size);
    // Glorot initialization
    double limit = sqrt(6.0 / (d_model + vocab_size));
    W = W * limit;

    b = RowVectorXd::Zero(vocab_size);
    
    d_W = MatrixXd(d_model, vocab_size);
    d_b = RowVectorXd(vocab_size);

    params = {TrainableParameter(W, d_W), TrainableParameter(b, d_b)};

    W_mlx = eigen_to_mlx(W);
}

void LinearLayer::forward(const MatrixXd& input) {
    // input : (num_tokens, d_model)
    X = input;
    output = (input * W).rowwise() + b;
}

void LinearLayer::forward_mlx(const mx::array& input) {
    // input : (num_tokens, d_model)
    X_mlx = input;
    output_mlx = mx::matmul(input, W_mlx) + b_mlx;
}

void LinearLayer::backward(const MatrixXd& d_output) {
    d_W += X.transpose() * d_output;
    d_b += d_output.colwise().sum();
    d_input = d_output * W.transpose();
}

void LinearLayer::backward_mlx(const mlx::core::array& d_output) {
    d_W_mlx = d_W_mlx + mx::matmul(mx::transpose(X_mlx), d_output);
    d_b_mlx = d_b_mlx + mx::sum(d_output, 0, true);
    d_input_mlx = mx::matmul(d_output, mx::transpose(W_mlx));
}

MatrixXd LinearLayer::infer(const MatrixXd& input) const {
    return (input * W).rowwise() + b;
}

mx::array LinearLayer::infer_mlx(const mx::array& input) const {
    return mx::matmul(input, W_mlx) + b_mlx;
}

const vector<TrainableParameter>& LinearLayer::get_parameters() const {
    return params;
}

const MatrixXd& LinearLayer::get_output() const {
    return output;
}

const mx::array& LinearLayer::get_output_mlx() const {
    return output_mlx;
}

const MatrixXd& LinearLayer::get_d_input() const {
    return d_input;
}

const mx::array& LinearLayer::get_d_input_mlx() const {
    return d_input_mlx;
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
    file << W << "\n";
    file << b << "\n";
}

void LinearLayer::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> d_model >> vocab_size;
    for (int i = 0; i < d_model; i++) {
        for (int j = 0; j < vocab_size; j++) {
            file >> W(i, j);
        }
    }
    for (int i = 0; i < vocab_size; i++) {
        file >> b(i);
    }
}