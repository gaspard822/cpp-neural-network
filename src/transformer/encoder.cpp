#include <iostream>
#include <fstream>
#include "transformer/encoder.hpp"
#include "core/relu.hpp"
#include "core/sigmoid.hpp"
#include "core/identity.hpp"

namespace mx = mlx::core;

Encoder::Encoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation) :
    seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), d_ff(d_ff), activation(activation),
    output(mlx::core::zeros({1, 1}, mx::float32)), d_input(mlx::core::zeros({1, 1}, mx::float32)) {

    ln1 = new LayerNorm(seq, d_model);
    mha_self = new MultiHeadAttention(seq, d_model, h, d_k, d_v, AttentionMode::ENCODER_SELF);
    ln2 = new LayerNorm(seq, d_model);
    ff = new FeedForward(activation, seq, d_model, d_ff);

    layers.push_back(ln1);
    layers.push_back(mha_self);
    layers.push_back(ln2);
    layers.push_back(ff);
}

void Encoder::forward(const mx::array& input) {
    ln1->forward_mlx(input);
    mha_self->forward_mlx(ln1->get_output_mlx());
    mx::array after_attn = mha_self->get_output_mlx() + input;

    ln2->forward_mlx(after_attn);
    ff->forward_mlx(ln2->get_output_mlx());
    output = ff->get_output_mlx() + after_attn;
}

void Encoder::backward(const mx::array& d_output) {
    ff->backward_mlx(d_output);
    ln2->backward_mlx(ff->get_d_input_mlx());
    mx::array d_after_attn = ln2->get_d_input_mlx() + d_output;

    mha_self->backward_mlx(d_after_attn);
    ln1->backward_mlx(mha_self->get_d_input_mlx());
    d_input = ln1->get_d_input_mlx() + d_after_attn;
}

mx::array Encoder::infer(const mx::array& input) {
    mx::array x_norm1 = ln1->infer_mlx(input);
    mx::array after_attn = mha_self->infer_mlx(x_norm1) + input;
    mx::array x_norm2 = ln2->infer_mlx(after_attn);
    return ff->infer_mlx(x_norm2) + after_attn;
}

const mx::array& Encoder::get_output() const {
    return output;
}

const mx::array& Encoder::get_d_input() const {
    return d_input;
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
