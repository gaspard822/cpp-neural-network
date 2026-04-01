#include <iostream>
#include <fstream>
#include <array>
#include "transformer/input_layer.hpp"
#include "transformer/bpe_tokenizer.hpp"
#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

InputLayer::InputLayer(int seq, int d_model, int vocab_size) : seq(seq), d_model(d_model), vocab_size(vocab_size),
                                                               embeddings_mlx(mx::zeros({vocab_size, d_model}, mx::float32)),
                                                               d_embeddings_mlx(mx::zeros({vocab_size, d_model}, mx::float32)),
                                                               positional_encodings_mlx(mx::zeros({seq, d_model}, mx::float32)) {
    // Initialize the embeddings matrix
    double limit = sqrt(6.0 / (vocab_size + d_model));
    embeddings = MatrixXd::Random(vocab_size, d_model) * limit;
    embeddings.row(BPETokenizer::PAD_ID).setZero();
    d_embeddings = MatrixXd::Zero(vocab_size, d_model);
    positional_encodings = compute_positional_encodings(seq, d_model);
    params = {TrainableParameter(embeddings, d_embeddings)};

    embeddings_mlx = eigen_to_mlx(embeddings);
    d_embeddings_mlx = mx::zeros({vocab_size, d_model}, mx::float32);
    positional_encodings_mlx = eigen_to_mlx(positional_encodings);
    output_mlx = mx::zeros({seq, d_model}, mx::float32);
}

MatrixXd InputLayer::compute_positional_encodings(int seq, int d_model) {
    MatrixXd P(seq, d_model);
    for (int pos = 0; pos < seq; ++pos) {
        for (int i = 0; i < d_model; i += 2) {
            double denom = pow(10000.0, static_cast<double>(i) / d_model);
            P(pos, i) = sin(pos / denom);
            if (i + 1 < d_model) {
                P(pos, i + 1) = cos(pos / denom);
            }
        }
    }
    return P;
}

void InputLayer::forward(const MatrixXd& input) {

}

void InputLayer::forward(const vector<int>& input_token_ids) {
    token_ids = input_token_ids;
    int num_tokens = input_token_ids.size();
    output = MatrixXd::Zero(num_tokens, d_model);
    for (int i = 0; i < num_tokens; i++) {
        output.row(i) = embeddings.row(input_token_ids[i]);
    }
    output += positional_encodings.topRows(output.rows());
}

void InputLayer::forward_mlx(const vector<int>& input_token_ids) {
    token_ids = input_token_ids;
    int num_tokens = input_token_ids.size();
    // Gather the tokens' embeddings
    mx::array indices = mx::array(input_token_ids.data(), {num_tokens}, mx::int32);
    output_mlx = mx::take(embeddings_mlx, indices, 0);
    // Add positional encodings
    output_mlx = output_mlx + mx::slice(positional_encodings_mlx, {0, 0}, {num_tokens, d_model});
    mx::eval(output_mlx);
}

void InputLayer::backward(const MatrixXd& d_output) {
    int num_tokens = token_ids.size();

    for (int i = 0; i < num_tokens; i++) {
        d_embeddings.row(token_ids[i]) += d_output.row(i);
    }
    d_embeddings.row(BPETokenizer::PAD_ID).setZero();
}

void InputLayer::backward_mlx(const mx::array& d_output) {
    mx::array indices = mx::array(token_ids.data(), {(int)token_ids.size()}, mx::int32);
    
    // Reshape to 3D because scatter_add requires updates.ndim = indices.ndim + a.ndim
    mx::array updates = mx::reshape(d_output, {(int)token_ids.size(), 1, d_model});
    d_embeddings_mlx = mx::scatter_add(d_embeddings_mlx, indices, updates, 0);

    // Zero out PAD row
    d_embeddings_mlx = mx::slice_update(d_embeddings_mlx, mx::zeros({1, d_model}, mx::float32), {BPETokenizer::PAD_ID, 0}, {BPETokenizer::PAD_ID + 1, d_model});
}

MatrixXd InputLayer::infer(const MatrixXd& layer_input) const {
    return MatrixXd();
}

MatrixXd InputLayer::infer(const vector<int>& input_token_ids) const {
    int num_tokens = input_token_ids.size();
    MatrixXd output_tmp = MatrixXd::Zero(num_tokens, d_model);
    for (int i = 0; i < num_tokens; i++) {
        output_tmp.row(i) = embeddings.row(input_token_ids[i]);
    }
    output_tmp += positional_encodings.topRows(output_tmp.rows());
    return output_tmp;
}

mlx::core::array InputLayer::infer_mlx(const vector<int>& input_token_ids) const {
    int num_tokens = input_token_ids.size();
    mx::array indices = mx::array(input_token_ids.data(), {num_tokens}, mx::int32);
    mx::array output_tmp_mlx = mx::take(embeddings_mlx, indices, 0);
    output_tmp_mlx = output_tmp_mlx + mx::slice(positional_encodings_mlx, {0, 0}, {num_tokens, d_model});
    mx::eval(output_tmp_mlx);
    return output_tmp_mlx;
}

const vector<TrainableParameter>& InputLayer::get_parameters() const {
    return params;
}

const MatrixXd& InputLayer::get_output() const {
    return output;
}

const mlx::core::array& InputLayer::get_output_mlx() const {
    return output_mlx;
}

const MatrixXd& InputLayer::get_d_input() const {
    return d_input;
}

string InputLayer::get_layer_name() const {
    return "InputLayer";
}

string InputLayer::get_activation_name() const {
    return "";
}

LayerType InputLayer::get_type() const {
    return LayerType::INPUT_LAYER;
}

void InputLayer::save(ofstream& file) const {
    file << get_layer_name() << "\n";
    file << seq << " " << d_model << " " << vocab_size << "\n";
    file << embeddings << "\n";
    file << positional_encodings << "\n";
}

void InputLayer::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model >> vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < d_model; j++) {
            file >> embeddings(i, j);
        }
    }
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < d_model; j++) {
            file >> positional_encodings(i, j);
        }
    }
}