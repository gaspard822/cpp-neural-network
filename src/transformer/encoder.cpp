#include <iostream>
#include "transformer/encoder.hpp"

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
    int i = 0;
    for (Layer* layer: layers) {
        cout << "Forwarding through layer " << i << " of the encoder." << endl;  // debug
        i++;
        layer->forward(*layer_output);
        layer_output = &layer->get_output();
    }
    output = *layer_output;
}

const MatrixXd& Encoder::get_output() const {
    return output;
}