#include <iostream>
#include "transformer/decoder.hpp"

Decoder::Decoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation) :
    seq(seq), d_model(d_model), h(h), d_k(d_k), d_v(d_v), d_ff(d_ff), activation(activation) {

    // LayerNorm
    layers.push_back(new LayerNorm(seq, d_model));
    // MultiHeadAttention
    layers.push_back(new MultiHeadAttention(seq, d_model, h, d_k, d_v, AttentionMode::DECODER_MASKED_SELF));
    // LayerNorm
    layers.push_back(new LayerNorm(seq, d_model));
    // MultiHeadAttention
    mha_cross = new MultiHeadAttention(seq, d_model, h, d_k, d_v, AttentionMode::DECODER_CROSS);
    layers.push_back(mha_cross);
    // LayerNorm
    layers.push_back(new LayerNorm(seq, d_model));
    // FeedForward
    layers.push_back(new FeedForward(activation, seq, d_model, d_ff));

}

void Decoder::forward(const MatrixXd& encoder_input, const MatrixXd& decoder_input) {
    mha_cross->set_encoder_output(encoder_input);
    const MatrixXd* layer_output = &decoder_input;
    for (Layer* layer: layers) {
        layer->forward(*layer_output);
        layer_output = &layer->get_output();
    }
    output = *layer_output;
}

void Decoder::backward(const MatrixXd& d_output) {
    const MatrixXd* layer_d_input = &d_output;
    int num_layers = layers.size();
    for (int i = num_layers - 1; i >= 0; i--) {
        layers[i]->backward(*layer_d_input);
        layer_d_input = &layers[i]->get_d_input();
    }
    d_input = *layer_d_input;
    d_encoder_input = mha_cross->get_d_encoder_output();
}

MatrixXd Decoder::infer(const MatrixXd& encoder_input, const MatrixXd& decoder_input) {
    mha_cross->set_encoder_output(encoder_input);
    MatrixXd output = decoder_input;
    for (Layer* layer : layers) {
        output = layer->infer(output);
    }
    return output;
}

const MatrixXd& Decoder::get_output() const {
    return output;
}

const MatrixXd& Decoder::get_d_input() const {
    return d_input;
}

const MatrixXd& Decoder::get_d_encoder_input() const {
    return d_encoder_input;
}

const vector<Layer*>& Decoder::get_layers() {
    return layers;
}
