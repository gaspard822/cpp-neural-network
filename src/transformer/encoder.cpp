#include <iostream>
#include <fstream>
#include "transformer/encoder.hpp"
#include "core/relu.hpp"
#include "core/sigmoid.hpp"
#include "core/identity.hpp"

Encoder::Encoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation) :
    seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), d_ff(d_ff), activation(activation) {

    // LayerNorm
    layers.push_back(new LayerNorm(seq, d_model));
    // MultiHeadAttention
    layers.push_back(new MultiHeadAttention(seq, d_model, h, d_k, d_v, AttentionMode::ENCODER_SELF));
    // LayerNorm
    layers.push_back(new LayerNorm(seq, d_model));
    // FeedForward
    layers.push_back(new FeedForward(activation, seq, d_model, d_ff));

}

void Encoder::forward(const MatrixXd& input) {
    const MatrixXd* layer_output = &input;
    for (Layer* layer: layers) {
        layer->forward(*layer_output);
        layer_output = &layer->get_output();
    }
}

void Encoder::backward(const MatrixXd& d_output) {
    const MatrixXd* layer_d_input = &d_output;
    int num_layers = layers.size();
    for (int i = num_layers - 1; i >= 0; i--) {
        layers[i]->backward(*layer_d_input);
        layer_d_input = &layers[i]->get_d_input();
    }
}

MatrixXd Encoder::infer(const MatrixXd& input) {
    MatrixXd output = input;
    for (Layer* layer : layers) {
        output = layer->infer(output);
    }
    return output;
}

const MatrixXd& Encoder::get_output() const {
    return layers.back()->get_output();
}

const MatrixXd& Encoder::get_d_input() const {
    return layers.front()->get_d_input();
}

const vector<Layer*>& Encoder::get_layers() {
    return layers;
}

void Encoder::save(ofstream& file) const {
    file << "Encoder\n";
    file << seq << " " << d_model << " " << h << " " <<  d_k << " " << d_v << " " << d_ff << "\n";
    file << activation->get_activation_name() << "\n";
    for (Layer* layer: layers) {
        layer->save(file);
    }
}

void Encoder::load(ifstream& file) {
    string block_type;
    file >> block_type;
    if (block_type != "Encoder") throw runtime_error("Wrong layer was given. Got " + block_type + ", expected Encoder");

    file >> seq >> d_model >> h >> d_k >> d_v >> d_ff;
    string activation_name;
    file >> activation_name;
    if (activation_name == "Relu") {
        activation = new Relu();
    } else if (activation_name == "Sigmoid") {
        activation = new Sigmoid();
    } else if (activation_name == "Identity") {
        activation = new Identity();
    } else {
        throw runtime_error("Couldn't read the type of the activation function. Got " + activation_name);
    }

    for (Layer* layer: layers) {
        layer->load(file);
    }
}
