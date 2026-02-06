#include "transformer/transformer.hpp"

Transformer::Transformer(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h) :
        num_encoder_layers(num_encoder_layers), num_decoder_layers(num_decoder_layers), seq(seq), d_model(d_model), h(h) {
    
    d_k = d_v = d_model / h;
    d_ff = 4 * d_model;

    
    for (int i = 0; i < num_encoder_layers; i++) {
        encoders.push_back(new Encoder(seq, d_model, h, d_k, d_v));
    }
    
    for (int i = 0; i < num_encoder_layers; i++) {
        decoders.push_back(new Decoder(seq, d_model, h, d_k, d_v));
    }
}