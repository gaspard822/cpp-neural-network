#include <iostream>
#include "transformer/transformer.hpp"

Transformer::Transformer(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h,
                         Tokenizer* tokenizer, ActivationFunction* activation, LossFunction* loss_function, Optimizer* optimizer) :
        num_encoder_layers(num_encoder_layers), num_decoder_layers(num_decoder_layers), seq(seq), d_model(d_model), h(h),
        tokenizer(tokenizer), activation(activation), loss_function(loss_function), optimizer(optimizer) {
    
    d_k = d_v = d_model / h;
    d_ff = 4 * d_model;

    encoder_input_layer = new InputLayer(seq, d_model, tokenizer);
    for (int i = 0; i < num_encoder_layers; i++) {
        encoders.push_back(new Encoder(seq, d_model, h, d_k, d_v, d_ff, activation));
    }

    decoder_input_layer = new InputLayer(seq, d_model, tokenizer);
    for (int i = 0; i < num_encoder_layers; i++) {
        decoders.push_back(new Decoder(seq, d_model, h, d_k, d_v, d_ff, activation));
    }

    linear_layer = new LinearLayer(d_model, tokenizer->get_vocab_size());
}

Transformer::~Transformer() {
    for (Encoder* encoder : encoders) {
        delete encoder;
    }
    for (Decoder* decoder : decoders) {
        delete decoder;
    }
    if (tokenizer) delete tokenizer;
    if (activation) delete activation;
    if (loss_function) delete loss_function;
    if (optimizer) delete optimizer;
}

MatrixXd Transformer::forward(const string& text) {
    cout << "vocab_size: " << tokenizer->get_vocab_size() << endl << endl;  // debug
    cout << "########## FORWARDING THROUGH THE ENCODER ##########" << endl;  // debug
    encoder_input_layer->forward(text);
    const MatrixXd* encoder_output = &encoder_input_layer->get_output();
    // Forward that embedding into the encoders
    for (Encoder* encoder: encoders) {
        encoder->forward(*encoder_output);
        encoder_output = &encoder->get_output();
    }

    cout << "########## FORWARDING THROUGH THE DECODER ##########" << endl;  // debug
    const MatrixXd* decoder_output = &decoder_input_layer->get_output();
    // Get the decoder's embeddings corresponding to the given text
    decoder_input_layer->forward(text);
    // Forward that embedding into the encoders
    for (Decoder* decoder: decoders) {
        decoder->forward(*encoder_output, *decoder_output);
        decoder_output = &decoder->get_output();
    }

    cout << "########## FORWARDING THROUGH THE LINEAR LAYER ##########" << endl;  // debug
    linear_layer->forward(*decoder_output);
    return linear_layer->get_output();
}

void Transformer::backward(const MatrixXd& y_true, const MatrixXd& y_pred) {
    /*
    // First compute the derivative of the loss with respect to the loss function
    MatrixXd d_loss_buf = loss_function->derivative(y_true, y_pred);
    const MatrixXd* d_loss = &d_loss_buf;
    // Propagate the gradients with respect to each layer back into the network and update the parameters accordingly
    int num_layers = layers.size();
    for (int i = num_layers - 1; i >= 0; i--) {
        layers[i]->backward(*d_loss);
        optimizer->update_parameters(i);
        d_loss = &layers[i]->get_d_input();
    }
    */
}