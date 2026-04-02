#include <fstream>
#include <iostream>
#include "transformer/decoder.hpp"
#include "core/relu.hpp"
#include "core/sigmoid.hpp"
#include "core/identity.hpp"

namespace mx = mlx::core;

Decoder::Decoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation) :
    seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), d_ff(d_ff), activation(activation),
    output(mlx::core::zeros({1, 1}, mx::float32)), d_input(mlx::core::zeros({1, 1}, mx::float32)), d_encoder_input(mlx::core::zeros({1, 1}, mx::float32)) {

    ln1 = new LayerNorm(seq, d_model);
    mha_masked = new MultiHeadAttention(seq, d_model, h, d_k, d_v, AttentionMode::DECODER_MASKED_SELF);
    ln2 = new LayerNorm(seq, d_model);
    mha_cross = new MultiHeadAttention(seq, d_model, h, d_k, d_v, AttentionMode::DECODER_CROSS);
    ln3 = new LayerNorm(seq, d_model);
    ff = new FeedForward(activation, seq, d_model, d_ff);

    layers.push_back(ln1);
    layers.push_back(mha_masked);
    layers.push_back(ln2);
    layers.push_back(mha_cross);
    layers.push_back(ln3);
    layers.push_back(ff);
}

void Decoder::forward(const mx::array& encoder_input, const mx::array& decoder_input) {
    ln1->forward_mlx(decoder_input);
    mha_masked->forward_mlx(ln1->get_output_mlx());
    mx::array after_masked = mha_masked->get_output_mlx() + decoder_input;

    ln2->forward_mlx(after_masked);
    mha_cross->set_encoder_output_mlx(encoder_input);
    mha_cross->forward_mlx(ln2->get_output_mlx());
    mx::array after_cross = mha_cross->get_output_mlx() + after_masked;

    ln3->forward_mlx(after_cross);
    ff->forward_mlx(ln3->get_output_mlx());
    output = ff->get_output_mlx() + after_cross;
}

void Decoder::backward(const mx::array& d_output) {
    ff->backward_mlx(d_output);
    ln3->backward_mlx(ff->get_d_input_mlx());
    mx::array d_after_cross = ln3->get_d_input_mlx() + d_output;

    mha_cross->backward_mlx(d_after_cross);
    d_encoder_input = mha_cross->get_d_encoder_output_mlx();
    ln2->backward_mlx(mha_cross->get_d_input_mlx());
    mx::array d_after_masked = ln2->get_d_input_mlx() + d_after_cross;

    mha_masked->backward_mlx(d_after_masked);
    ln1->backward_mlx(mha_masked->get_d_input_mlx());
    d_input = ln1->get_d_input_mlx() + d_after_masked;
}

mx::array Decoder::infer(const mx::array& encoder_input, const mx::array& decoder_input) {
    mha_cross->set_encoder_output_mlx(encoder_input);
    mx::array x_norm1 = ln1->infer_mlx(decoder_input);
    mx::array after_masked = mha_masked->infer_mlx(x_norm1) + decoder_input;
    mx::array x_norm2 = ln2->infer_mlx(after_masked);
    mx::array after_cross = mha_cross->infer_mlx(x_norm2) + after_masked;
    mx::array x_norm3 = ln3->infer_mlx(after_cross);
    return ff->infer_mlx(x_norm3) + after_cross;
}

const mx::array& Decoder::get_output() const {
    return output;
}

const mx::array& Decoder::get_d_input() const {
    return d_input;
}

const mx::array& Decoder::get_d_encoder_input() const {
    return d_encoder_input;
}

const vector<Layer*>& Decoder::get_layers() {
    return layers;
}

void Decoder::save(ofstream& file) const {
    file << "Decoder\n";
    file << seq << " " << d_model << " " << h << " " <<  d_k << " " << d_v << " " << d_ff << "\n";
    file << activation->get_activation_name() << "\n";
    for (Layer* layer: layers) {
        layer->save(file);
    }
}

void Decoder::load(ifstream& file) {
    string block_type;
    file >> block_type;
    if (block_type != "Decoder") throw runtime_error("Wrong layer was given. Got " + block_type + ", expected Decoder");

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
