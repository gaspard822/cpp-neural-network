#include <iostream>
#include <fstream>
#include <array>
#include "transformer/input_layer.hpp"
#include "transformer/bpe_tokenizer.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

InputLayer::InputLayer(int seq, int d_model, int vocab_size) : seq(seq), d_model(d_model), vocab_size(vocab_size),
                                                               token_ids(mx::zeros({1, 1}, mx::float32)),
                                                               embeddings(mx::zeros({vocab_size, d_model}, mx::float32)),
                                                               d_embeddings(mx::zeros({vocab_size, d_model}, mx::float32)),
                                                               positional_encodings(mx::zeros({seq, d_model}, mx::float32)) {
    // Initialize the embeddings matrix
    float glorot_factor = sqrt(6.0f / (vocab_size + d_model));
    embeddings = mx::random::uniform(-glorot_factor, glorot_factor, {vocab_size, d_model});
    embeddings = mx::slice_update(embeddings, mx::zeros({1, d_model}, mx::float32), {BPETokenizer::PAD_ID, 0}, {BPETokenizer::PAD_ID + 1, d_model});
    positional_encodings = compute_positional_encodings(seq, d_model);
    params = {TrainableParameter(embeddings, d_embeddings)};
}

mx::array InputLayer::compute_positional_encodings(int seq, int d_model) {
    vector<float> P(seq * d_model, 0.0f);
    for (int pos = 0; pos < seq; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            float denom = pow(10000.0f, static_cast<float>(i) / d_model);
            P[pos * d_model + i] = sin(pos / denom);
            if (i + 1 < d_model) {
                P[pos * d_model + i + 1] = cos(pos / denom);
            }
        }
    }
    return mx::array(P.begin(), {seq, d_model}, mx::float32);
}

void InputLayer::forward(const mx::array& input_token_ids) {
    // input_token_ids has shape {num_sentences, max_sentence_length}
    token_ids = input_token_ids;
    int num_sentences = input_token_ids.shape(0);
    int max_sentence_length = input_token_ids.shape(1);
    mx::array indices = mx::reshape(input_token_ids, {num_sentences * max_sentence_length});
    output = mx::take(embeddings, indices, 0);  // output has shape {num_sentences * max_sentence_length, d_model}
    output = mx::reshape(output, {num_sentences, max_sentence_length, d_model});  // output has shape {num_sentences, max_sentence_length, d_model}
    output = output + mx::slice(positional_encodings, {0, 0}, {max_sentence_length, d_model});
}

void InputLayer::backward(const mx::array& d_output) {
    // d_output has shape {num_sentences, max_sentence_length, d_model}
    int num_sentences = d_output.shape(0);
    int max_sentence_length = d_output.shape(1);
    mx::array indices = mx::reshape(token_ids, {num_sentences * max_sentence_length});
    mx::array updates = mx::reshape(d_output, {num_sentences * max_sentence_length, 1, d_model});
    d_embeddings = mx::scatter_add(mx::zeros({vocab_size, d_model}, mx::float32), indices, updates, 0);

    // Zero out PAD row
    d_embeddings = mx::slice_update(d_embeddings, mx::zeros({1, d_model}, mx::float32), {BPETokenizer::PAD_ID, 0}, {BPETokenizer::PAD_ID + 1, d_model});
}

mx::array InputLayer::infer(const mx::array& layer_input) const {
    return mx::zeros({1, 1}, mx::float32);
}

const vector<TrainableParameter>& InputLayer::get_parameters() const {
    return params;
}

const mx::array& InputLayer::get_output() const {
    return output;
}

const mx::array& InputLayer::get_d_input() const {
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
    save_array(file, embeddings);
    save_array(file, positional_encodings);
}

void InputLayer::load(ifstream& file) {
    string layer_name;
    file >> layer_name;
    if (layer_name != get_layer_name()) throw runtime_error("Wrong layer was given. Got " + layer_name + ", expected " + get_layer_name());

    file >> seq >> d_model >> vocab_size;
    embeddings = load_array(file);
    positional_encodings = load_array(file);
}
