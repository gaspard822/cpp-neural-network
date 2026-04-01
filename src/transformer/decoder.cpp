#include <fstream>
#include <iostream>
#include "transformer/decoder.hpp"
#include "core/relu.hpp"
#include "core/sigmoid.hpp"
#include "core/identity.hpp"

namespace mx = mlx::core;

Decoder::Decoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation) :
    seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), d_ff(d_ff), activation(activation),
    output_mlx(mlx::core::zeros({1, 1}, mx::float32)), d_input_mlx(mlx::core::zeros({1, 1}, mx::float32)), d_encoder_input_mlx(mlx::core::zeros({1, 1}, mx::float32)) {

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

void Decoder::forward(const MatrixXd& encoder_input, const MatrixXd& decoder_input) {
    ln1->forward(decoder_input);
    mha_masked->forward(ln1->get_output());
    MatrixXd after_masked = mha_masked->get_output() + decoder_input;

    ln2->forward(after_masked);
    mha_cross->set_encoder_output(encoder_input);
    mha_cross->forward(ln2->get_output());
    MatrixXd after_cross = mha_cross->get_output() + after_masked;

    ln3->forward(after_cross);
    ff->forward(ln3->get_output());
    output = ff->get_output() + after_cross;
}

void Decoder::forward_mlx(const mx::array& encoder_input, const mx::array& decoder_input) {
    ln1->forward_mlx(decoder_input);
    mha_masked->forward_mlx(ln1->get_output_mlx());
    mx::array after_masked = mha_masked->get_output_mlx() + decoder_input;

    ln2->forward_mlx(after_masked);
    mha_cross->set_encoder_output_mlx(encoder_input);
    mha_cross->forward_mlx(ln2->get_output_mlx());
    mx::array after_cross = mha_cross->get_output_mlx() + after_masked;

    ln3->forward_mlx(after_cross);
    ff->forward_mlx(ln3->get_output_mlx());
    output_mlx = ff->get_output_mlx() + after_cross;
}

void Decoder::backward(const MatrixXd& d_output) {
    ff->backward(d_output);
    ln3->backward(ff->get_d_input());
    MatrixXd d_after_cross = ln3->get_d_input() + d_output;

    mha_cross->backward(d_after_cross);
    d_encoder_input = mha_cross->get_d_encoder_output();
    ln2->backward(mha_cross->get_d_input());
    MatrixXd d_after_masked = ln2->get_d_input() + d_after_cross;

    mha_masked->backward(d_after_masked);
    ln1->backward(mha_masked->get_d_input());
    d_input = ln1->get_d_input() + d_after_masked;
}

void Decoder::backward_mlx(const mx::array& d_output) {
    ff->backward_mlx(d_output);
    ln3->backward_mlx(ff->get_d_input_mlx());
    mx::array d_after_cross = ln3->get_d_input_mlx() + d_output;

    mha_cross->backward_mlx(d_after_cross);
    d_encoder_input_mlx = mha_cross->get_d_encoder_output_mlx();
    ln2->backward_mlx(mha_cross->get_d_input_mlx());
    mx::array d_after_masked = ln2->get_d_input_mlx() + d_after_cross;

    mha_masked->backward_mlx(d_after_masked);
    ln1->backward_mlx(mha_masked->get_d_input_mlx());
    d_input_mlx = ln1->get_d_input_mlx() + d_after_masked;
}

MatrixXd Decoder::infer(const MatrixXd& encoder_input, const MatrixXd& decoder_input) {
    mha_cross->set_encoder_output(encoder_input);
    MatrixXd x_norm1 = ln1->infer(decoder_input);
    MatrixXd after_masked = mha_masked->infer(x_norm1) + decoder_input;
    MatrixXd x_norm2 = ln2->infer(after_masked);
    MatrixXd after_cross = mha_cross->infer(x_norm2) + after_masked;
    MatrixXd x_norm3 = ln3->infer(after_cross);
    return ff->infer(x_norm3) + after_cross;
}

const MatrixXd& Decoder::get_output() const {
    return output;
}

const mx::array& Decoder::get_output_mlx() const {
    return output_mlx;
}

const MatrixXd& Decoder::get_d_input() const {
    return d_input;
}

const mx::array& Decoder::get_d_input_mlx() const {
    return d_input_mlx;
}

const MatrixXd& Decoder::get_d_encoder_input() const {
    return d_encoder_input;
}

const mx::array& Decoder::get_d_encoder_input_mlx() const {
    return d_encoder_input_mlx;
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