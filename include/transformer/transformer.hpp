#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "transformer/input_layer.hpp"
#include "transformer/encoder.hpp"
#include "transformer/decoder.hpp"

class Transformer {
    private:
        int num_encoder_layers, num_decoder_layers;
        int seq, d_model, h;
        int d_k, d_v;  // We simply set d_k and d_v to d_model / h
        int d_ff;  // We simply set d_ff to 4 * d_model

        InputLayer* encoder_input;
        vector<Encoder*> encoders;

        InputLayer* decoder_input;
        vector<Decoder*> decoders;
        
    public:
        Transformer(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h);

        void train();
        
};

#endif