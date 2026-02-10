#include <iostream>
#include "transformer/transformer.hpp"

TransformerNetwork::TransformerNetwork(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h,
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

TransformerNetwork::~TransformerNetwork() {
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

MatrixXd TransformerNetwork::forward(const string& encoder_text, const string& decoder_text) {
    cout << "vocab_size: " << tokenizer->get_vocab_size() << endl << endl;  // debug
    cout << "########## FORWARDING THROUGH THE ENCODERS ##########" << endl;  // debug
    encoder_input_layer->forward(encoder_text);
    const MatrixXd* encoder_output = &encoder_input_layer->get_output();
    // Forward that embedding into the encoders
    for (Encoder* encoder: encoders) {
        encoder->forward(*encoder_output);
        encoder_output = &encoder->get_output();
    }

    cout << "########## FORWARDING THROUGH THE DECODERS ##########" << endl;  // debug
    const MatrixXd* decoder_output = &decoder_input_layer->get_output();
    // Get the decoder's embeddings corresponding to the given text
    decoder_input_layer->forward(decoder_text);
    // Forward that embedding into the encoders
    for (Decoder* decoder: decoders) {
        decoder->forward(*encoder_output, *decoder_output);
        decoder_output = &decoder->get_output();
    }

    cout << "########## FORWARDING THROUGH THE LINEAR LAYER ##########" << endl;  // debug
    linear_layer->forward(*decoder_output);
    return linear_layer->get_output();
}

void TransformerNetwork::backward(const MatrixXd& y_true, const MatrixXd& y_pred) {
    MatrixXd d_loss = loss_function->derivative(y_true, y_pred);
    const MatrixXd* decoder_d_input = &d_loss;
    cout << "########## BACKWARDING THROUGH THE LINEAR LAYER ##########" << endl;  // debug
    linear_layer->backward(*decoder_d_input);
    decoder_d_input = &linear_layer->get_d_input();

    cout << "########## BACKWARDING THROUGH THE DECODERS ##########" << endl;  // debug
    MatrixXd encoder_d_input_buf = MatrixXd::Zero(seq, d_model);
    for (int i = num_decoder_layers - 1; i >= 0; i--) {
        decoders[i]->backward(*decoder_d_input);
        decoder_d_input = &decoders[i]->get_d_input();
        encoder_d_input_buf += decoders[i]->get_d_encoder_input();
    }
    decoder_input_layer->backward(*decoder_d_input);

    cout << "########## BACKWARDING THROUGH THE ENCODERS ##########" << endl;  // debug
    const MatrixXd* encoder_d_input = &encoder_d_input_buf;
    for (int i = num_encoder_layers - 1; i >= 0; i--) {
        encoders[i]->backward(*encoder_d_input);
        encoder_d_input = &decoders[i]->get_d_input();
    }
    encoder_input_layer->backward(*encoder_d_input);

    cout << "########## UPDATING THE PARAMETERS ##########" << endl;  // debug
    optimizer->update_parameters();
}

void TransformerNetwork::save_model(const string& path) const {
    // TODO
}

void TransformerNetwork::load_model(const string& filename) {
    // TODO
}

Optimizer* TransformerNetwork::get_optimizer() const {
    return optimizer;
}

NetworkType TransformerNetwork::get_type() const {
    return NetworkType::TRANSFORMER;
}

