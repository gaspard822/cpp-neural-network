#include <iostream>
#include "transformer/transformer.hpp"
#include "transformer/tokenizer.hpp"

TransformerNetwork::TransformerNetwork(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h, int vocab_size,
                                       ActivationFunction* activation, CrossEntropy* cross_entropy_loss, Optimizer* optimizer) :
        num_encoder_layers(num_encoder_layers), num_decoder_layers(num_decoder_layers), seq(seq), d_model(d_model), h(h),
        vocab_size(vocab_size), activation(activation), cross_entropy_loss(cross_entropy_loss), optimizer(optimizer) {
    
    d_k = d_v = d_model / h;
    d_ff = 4 * d_model;

    layers = {};
    encoder_input_layer = new InputLayer(seq, d_model, vocab_size);
    layers.push_back((Layer*) encoder_input_layer);
    for (int i = 0; i < num_encoder_layers; i++) {
        Encoder* encoder = new Encoder(seq, d_model, h, d_k, d_v, d_ff, activation);
        encoders.push_back(encoder);
        layers.insert(layers.end(), encoder->get_layers().begin(), encoder->get_layers().end());
    }

    decoder_input_layer = new InputLayer(seq, d_model, vocab_size);
    for (int i = 0; i < num_decoder_layers; i++) {
        Decoder* decoder = new Decoder(seq, d_model, h, d_k, d_v, d_ff, activation);
        decoders.push_back(decoder);
        layers.insert(layers.end(), decoder->get_layers().begin(), decoder->get_layers().end());
    }

    linear_layer = new LinearLayer(d_model, vocab_size);
    layers.push_back((Layer*) linear_layer);

    optimizer->set_network(this);
}

TransformerNetwork::~TransformerNetwork() {
    for (Encoder* encoder : encoders) {
        delete encoder;
    }
    for (Decoder* decoder : decoders) {
        delete decoder;
    }
    if (activation) delete activation;
    if (cross_entropy_loss) delete cross_entropy_loss;
    if (optimizer) delete optimizer;
}

MatrixXd TransformerNetwork::forward(const vector<int>& encoder_token_ids, const vector<int>& decoder_token_ids) {
    encoder_input_layer->forward(encoder_token_ids);
    const MatrixXd* encoder_output = &encoder_input_layer->get_output();
    // Forward that embedding into the encoders
    for (Encoder* encoder: encoders) {
        encoder->forward(*encoder_output);
        encoder_output = &encoder->get_output();
    }

    decoder_input_layer->forward(decoder_token_ids);
    const MatrixXd* decoder_output = &decoder_input_layer->get_output();
    // Get the decoder's embeddings corresponding to the given text
    // Forward that embedding into the encoders
    for (Decoder* decoder: decoders) {
        decoder->forward(*encoder_output, *decoder_output);
        decoder_output = &decoder->get_output();
    }

    linear_layer->forward(*decoder_output);
    return linear_layer->get_output();
}

void TransformerNetwork::backward(const vector<int>& y_true, const MatrixXd& y_pred) {
    MatrixXd d_loss = cross_entropy_loss->derivative(y_true, y_pred);
    const MatrixXd* decoder_d_input = &d_loss;
    linear_layer->backward(*decoder_d_input);
    decoder_d_input = &linear_layer->get_d_input();

    int num_encoder_tokens = encoders[encoders.size()-1]->get_output().rows();
    MatrixXd encoder_d_input_buf = MatrixXd::Zero(num_encoder_tokens, d_model);
    for (int i = num_decoder_layers - 1; i >= 0; i--) {
        decoders[i]->backward(*decoder_d_input);
        decoder_d_input = &decoders[i]->get_d_input();
        encoder_d_input_buf += decoders[i]->get_d_encoder_input();
    }
    decoder_input_layer->backward(*decoder_d_input);

    const MatrixXd* encoder_d_input = &encoder_d_input_buf;
    for (int i = num_encoder_layers - 1; i >= 0; i--) {
        encoders[i]->backward(*encoder_d_input);
        encoder_d_input = &encoders[i]->get_d_input();
    }
    encoder_input_layer->backward(*encoder_d_input);
}

void TransformerNetwork::reset_gradients() {
    for (Layer* layer: layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            auto gradients = p.grad().setZero();
        }
    }
}

void TransformerNetwork::normalize_gradients(int batch_size) {
    for (Layer* layer: layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            auto gradients = p.grad();
            gradients /= batch_size;
        }
    }
}

void TransformerNetwork::train(vector<vector<int>>& encoder_sentences, vector<vector<int>>& decoder_sentences, int batch_size) {
    int num_sentences = encoder_sentences.size();
    int skipped_sentences = 0;
    reset_gradients();
    for (int i = 0; i < 1; i++) {
        if ((encoder_sentences[i].size() > seq) || (decoder_sentences[i].size() > seq)) {
            skipped_sentences++;
            continue;
        }
        vector<int> encoder_token_ids = encoder_sentences[i];
        vector<int> decoder_input_token_ids(decoder_sentences[i].begin(), decoder_sentences[i].end() - 1);
        vector<int> decoder_target_token_ids(decoder_sentences[i].begin() + 1, decoder_sentences[i].end());
        MatrixXd forward_X_batch = forward(encoder_token_ids, decoder_input_token_ids);
        backward(decoder_target_token_ids, forward_X_batch);
        
        if (i % batch_size == 0) {
            double loss = cross_entropy_loss->compute(decoder_target_token_ids, forward_X_batch);
            normalize_gradients(batch_size);
            optimizer->update_parameters();
            reset_gradients();
        }
    }
    cout << "Skipped " << skipped_sentences << " sentences" << endl;
}

vector<int> TransformerNetwork::infer(const vector<int>& encoder_token_ids) const {
    encoder_input_layer->forward(encoder_token_ids);
    const MatrixXd* encoder_output = &encoder_input_layer->get_output();
    // Forward that embedding into the encoders
    for (Encoder* encoder: encoders) {
        encoder->forward(*encoder_output);
        encoder_output = &encoder->get_output();
    }

    int last_token = Tokenizer::SOS_ID;
    vector<int> predicted_tokens = {last_token};
    int max_size = 100;
    while ((last_token != Tokenizer::EOS_ID) && (predicted_tokens.size() < max_size)) {
        decoder_input_layer->forward(predicted_tokens);
        const MatrixXd* decoder_output = &decoder_input_layer->get_output();
        // Get the decoder's embeddings corresponding to the given text
        // Forward that embedding into the encoders
        for (Decoder* decoder: decoders) {
            decoder->forward(*encoder_output, *decoder_output);
            decoder_output = &decoder->get_output();
        }

        linear_layer->forward(*decoder_output);
        MatrixXd linear_layer_output = linear_layer->get_output();
        int last_row = linear_layer_output.rows() - 1;
        linear_layer_output.row(last_row).maxCoeff(&last_token);
        predicted_tokens.push_back(last_token);
    }
    return predicted_tokens;
}

void TransformerNetwork::save_model(const string& path) const {
    // TODO
}

void TransformerNetwork::load_model(const string& filename) {
    // TODO
}

const vector<Layer*>& TransformerNetwork::get_layers() const {
    return layers;
}

Optimizer* TransformerNetwork::get_optimizer() const {
    return optimizer;
}

NetworkType TransformerNetwork::get_type() const {
    return NetworkType::TRANSFORMER;
}

