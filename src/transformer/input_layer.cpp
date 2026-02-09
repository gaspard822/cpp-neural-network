#include <iostream>
#include <fstream>
#include <array>
#include "transformer/input_layer.hpp"

InputLayer::InputLayer(int seq, int d_model, Tokenizer* tokenizer) : seq(seq), d_model(d_model), tokenizer(tokenizer) {
    vocab_size = tokenizer->get_vocab_size();
    // Initialize the embeddings matrix
    double limit = sqrt(6.0 / (vocab_size + d_model));
    embeddings = MatrixXd::Random(vocab_size, d_model) * limit;
    embeddings.row(Tokenizer::PAD_ID).setZero();
    d_embeddings = MatrixXd::Zero(vocab_size, d_model);
    positional_encodings = compute_positional_encodings(seq, d_model);
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

void InputLayer::forward(const string& text) {
    cout << "========== InputLayer::forward() ==========" << endl;  // debug
    token_ids = tokenizer->encode(text);
    cout << "token_ids:" << endl;
    for (auto it = token_ids.begin(); it != token_ids.end(); it++) {  // debug
        cout << *it << ", ";
    }
    cout << endl;
    int num_tokens = token_ids.size();
    output = MatrixXd::Zero(num_tokens, d_model);
    for (int i = 0; i < num_tokens; i++) {
        output.row(i) = embeddings.row(token_ids[i]);
    }
    output += positional_encodings.topRows(output.rows());
    cout << "+++ output (" << output.rows() << "," << output.cols() << "):" << endl << output << endl << endl; // debug
}

void InputLayer::backward(const MatrixXd& d_output) {
    int num_tokens = token_ids.size();

    d_embeddings.setZero();  // FIXME: This might be expensive if we use fancier tokenizers that have large vocab_size
    for (int i = 0; i < num_tokens; i++) {
        d_embeddings.row(token_ids[i]) = d_output.row(i);
    }
    d_embeddings.row(Tokenizer::PAD_ID).setZero();
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
