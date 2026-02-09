#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "transformer/input_layer.hpp"
#include "transformer/encoder.hpp"
#include "transformer/decoder.hpp"
#include "transformer/linear_layer.hpp"
#include "core/loss_function.hpp"
#include "core/optimizer.hpp"

class Transformer {
    private:
        int num_encoder_layers, num_decoder_layers;
        int seq, d_model, h;
        int d_k, d_v;  // We simply set d_k and d_v to d_model / h
        int d_ff;  // We simply set d_ff to 4 * d_model

        Tokenizer* tokenizer;
        ActivationFunction* activation;
        LossFunction* loss_function;
        Optimizer* optimizer;

        InputLayer* encoder_input_layer;
        vector<Encoder*> encoders;

        InputLayer* decoder_input_layer;
        vector<Decoder*> decoders;

        LinearLayer* linear_layer;
        
    public:
        Transformer(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h,
                    Tokenizer* tokenizer, ActivationFunction* activation, LossFunction* loss_function, Optimizer* optimizer);
        ~Transformer();

        MatrixXd forward(const string& text);
        void backward(const MatrixXd& y_true, const MatrixXd& y_pred);
        void train();
        
};

#endif