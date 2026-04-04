#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include "transformer/transformer.hpp"
#include "transformer/bpe_tokenizer.hpp"
#include "core/mlx_utils.hpp"

namespace mx = mlx::core;

void TransformerNetwork::init_layers() {
    d_k = d_v = d_model / h;
    d_ff = 4 * d_model;

    layers = {};
    encoder_input_layer = new InputLayer(seq, d_model, vocab_size);
    layers.push_back(encoder_input_layer);
    for (int i = 0; i < num_encoder_layers; i++) {
        Encoder* encoder = new Encoder(seq, d_model, h, d_k, d_v, d_ff, activation);
        encoders.push_back(encoder);
        layers.insert(layers.end(), encoder->get_layers().begin(), encoder->get_layers().end());
    }

    decoder_input_layer = new InputLayer(seq, d_model, vocab_size);
    layers.push_back(decoder_input_layer);
    for (int i = 0; i < num_decoder_layers; i++) {
        Decoder* decoder = new Decoder(seq, d_model, h, d_k, d_v, d_ff, activation);
        decoders.push_back(decoder);
        layers.insert(layers.end(), decoder->get_layers().begin(), decoder->get_layers().end());
    }

    linear_layer = new LinearLayer(d_model, vocab_size);
    layers.push_back(linear_layer);

    cross_entropy_loss = new CrossEntropy();
    optimizer->set_network(this);
}

TransformerNetwork::TransformerNetwork(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h, int vocab_size,
                                       ActivationFunction* activation, Optimizer* optimizer):
        num_encoder_layers(num_encoder_layers), num_decoder_layers(num_decoder_layers), seq(seq), d_model(d_model), h(h),
        vocab_size(vocab_size), activation(activation), optimizer(optimizer) {

    init_layers();
}

TransformerNetwork::TransformerNetwork(const string& path, ActivationFunction* activation, Optimizer* optimizer) {
    // Read architecture parameters from file
    ifstream file(path);
    if (!file) throw runtime_error("Can not open file: " + path);

    file >> num_encoder_layers >> num_decoder_layers >> seq >> d_model >> h >> vocab_size;
    file.close();

    // Set activation and optimizer
    this->activation = activation;
    this->optimizer = optimizer;

    // Create all layers
    init_layers();

    // Load weights from file
    load_model(path);
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

pair<vector<vector<int>>, vector<vector<int>>> preprocess_decoder_token_ids(const vector<vector<int>>& decoder_token_ids) {
    int num_sentences = decoder_token_ids.size();
    vector<vector<int>> decoder_input_token_ids;
    vector<vector<int>> decoder_target_token_ids;
    decoder_input_token_ids.reserve(num_sentences);
    decoder_target_token_ids.reserve(num_sentences);
    for (int i = 0; i < num_sentences; i++) {
        const vector<int> decoder_input_token_ids_tmp(decoder_token_ids[i].begin(), decoder_token_ids[i].end() - 1);
        const vector<int> decoder_target_token_ids_tmp(decoder_token_ids[i].begin() + 1, decoder_token_ids[i].end());
        decoder_input_token_ids.push_back(decoder_input_token_ids_tmp);
        decoder_target_token_ids.push_back(decoder_target_token_ids_tmp);
    }
    return {decoder_input_token_ids, decoder_target_token_ids};
}

const mx::array& TransformerNetwork::forward(const vector<int>& encoder_token_ids, const vector<int>& decoder_token_ids) {
    encoder_input_layer->forward(encoder_token_ids);
    const mx::array* encoder_output = &encoder_input_layer->get_output();
    // Forward that embedding into the encoders
    for (Encoder* encoder: encoders) {
        encoder->forward(*encoder_output);
        encoder_output = &encoder->get_output();
    }
    
    decoder_input_layer->forward(decoder_token_ids);
    const mx::array* decoder_output = &decoder_input_layer->get_output();
    // Forward that embedding into the decoders
    for (Decoder* decoder: decoders) {
        decoder->forward(*encoder_output, *decoder_output);
        decoder_output = &decoder->get_output();
    }
    
    linear_layer->forward(*decoder_output);
    return linear_layer->get_output();
}

const mx::array& TransformerNetwork::backward(const vector<int>& y_true, const mx::array& y_pred) {
    mx::array d_loss = cross_entropy_loss->derivative(y_true, y_pred);
    const mx::array* decoder_d_input = &d_loss;
    linear_layer->backward(*decoder_d_input);
    decoder_d_input = &linear_layer->get_d_input();

    int num_encoder_tokens = encoders[encoders.size()-1]->get_output().shape(0);
    mx::array encoder_d_input_buf = mx::zeros({num_encoder_tokens, d_model}, mx::float32);
    for (int i = num_decoder_layers - 1; i >= 0; i--) {
        decoders[i]->backward(*decoder_d_input);
        decoder_d_input = &decoders[i]->get_d_input();
        encoder_d_input_buf = encoder_d_input_buf + decoders[i]->get_d_encoder_input();
    }
    decoder_input_layer->backward(*decoder_d_input);

    const mx::array* encoder_d_input = &encoder_d_input_buf;
    for (int i = num_encoder_layers - 1; i >= 0; i--) {
        encoders[i]->backward(*encoder_d_input);
        encoder_d_input = &encoders[i]->get_d_input();
    }
    encoder_input_layer->backward(*encoder_d_input);
    return *encoder_d_input;
}

void TransformerNetwork::infer(const vector<vector<int>>& encoder_token_ids, BPETokenizer* tokenizer, const string& csv_path) const {
    ofstream file(csv_path);
    if (!file.is_open()) throw runtime_error("Could not open the file");

    int N = encoder_token_ids.size();
    for (int i = 0; i < N; i++) {
        mx::array encoder_output = encoder_input_layer->infer(encoder_token_ids[i]);
        // Forward that embedding into the encoders
        for (Encoder* encoder: encoders) {
            encoder_output = encoder->infer(encoder_output);
        }
    
        int last_token = BPETokenizer::SOS_ID;
        vector<int> predicted_tokens = {last_token};
        while ((last_token != BPETokenizer::EOS_ID) && (predicted_tokens.size() < seq)) {
            mx::array decoder_output = decoder_input_layer->infer(predicted_tokens);
            // Forward that embedding into the decoders
            for (Decoder* decoder: decoders) {
                decoder_output = decoder->infer(encoder_output, decoder_output);
            }
    
            mx::array linear_layer_output = linear_layer->infer(decoder_output);
            int last_row = linear_layer_output.shape(0) - 1;
            mx::array last_row_probs = mx::slice(linear_layer_output, {last_row, 0}, {last_row + 1, linear_layer_output.shape(1)});
            last_token = mx::argmax(last_row_probs, 1).item<int>();
            predicted_tokens.push_back(last_token);
        }
        string input_sentence = tokenizer->decode(encoder_token_ids[i]);
        string predicted_sentence = tokenizer->decode(predicted_tokens);
        file << "\"" << input_sentence << "\"," << "\"" << predicted_sentence << "\"\n";
    }
    file.close();
}

void TransformerNetwork::infer_live(BPETokenizer* tokenizer) const {
    cout << "Type a sentence in English and press Enter. Ctrl+C to quit." << endl;
    string line;
    while (true) {
        cout << "\n> ";
        if (!getline(cin, line)) break;
        if (line.empty()) continue;

        vector<int> encoder_token_ids = tokenizer->encode(line, true, true);

        mx::array encoder_output = encoder_input_layer->infer(encoder_token_ids);
        for (Encoder* encoder : encoders) {
            encoder_output = encoder->infer(encoder_output);
        }

        int last_token = BPETokenizer::SOS_ID;
        vector<int> predicted_tokens = {last_token};
        while (last_token != BPETokenizer::EOS_ID && predicted_tokens.size() < seq) {
            mx::array decoder_output = decoder_input_layer->infer(predicted_tokens);
            for (Decoder* decoder : decoders) {
                decoder_output = decoder->infer(encoder_output, decoder_output);
            }
            mx::array linear_layer_output = linear_layer->infer(decoder_output);
            int last_row = linear_layer_output.shape(0) - 1;
            mx::array last_row_probs = mx::slice(linear_layer_output, {last_row, 0}, {last_row + 1, linear_layer_output.shape(1)});
            last_token = mx::argmax(last_row_probs, 1).item<int>();
            predicted_tokens.push_back(last_token);
        }

        string predicted_sentence = tokenizer->decode(predicted_tokens);
        cout << predicted_sentence << endl;
    }
}

float TransformerNetwork::compute_validation_loss(vector<vector<int>>& encoder_tokens_val, vector<vector<int>>& decoder_tokens_val) {
    int N = encoder_tokens_val.size();
    float loss = 0.0f;
    for (int i = 0; i < N; i++) {
        const vector<int> decoder_input_token_ids(decoder_tokens_val[i].begin(), decoder_tokens_val[i].end() - 1);
        const vector<int> decoder_target_token_ids(decoder_tokens_val[i].begin() + 1, decoder_tokens_val[i].end());
        mx::array forwarded_sentence = forward(encoder_tokens_val[i], decoder_input_token_ids);
        loss += cross_entropy_loss->compute(decoder_target_token_ids, forwarded_sentence);
    }
    return loss / N;
}

void TransformerNetwork::reset_gradients() {
    for (Layer* layer: layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            *p.grad = mx::zeros_like(*p.grad);
        }
    }
}

void TransformerNetwork::normalize_gradients(int batch_size) {
    for (Layer* layer: layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            *p.grad = *p.grad / batch_size;
        }
    }
}

void TransformerNetwork::train(
    vector<vector<int>>& encoder_tokens_train, vector<vector<int>>& decoder_tokens_train,
    vector<vector<int>>& encoder_tokens_val, vector<vector<int>>& decoder_tokens_val,
    int epochs, int batch_size) {

    auto [decoder_input_token_ids_train, decoder_target_token_ids_train] = preprocess_decoder_token_ids(decoder_tokens_train);

    reset_gradients();
    for (int epoch = 0; epoch < epochs; epoch++) {
        cout << "Epoch " << epoch << endl;

        int N = encoder_tokens_train.size();

        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);  // 0, 1, 2, ..., N-1
        random_device rd;
        mt19937 gen(rd());
        shuffle(indices.begin(), indices.end(), gen);

        int count = 0;
        for (int start = 0; start < N; start += batch_size) {
            int end = min(start + batch_size, N);
            for (int i = start; i < end; i++) {
                if (count % 10000 == 0) cout << "Trained on " << count << " samples" << endl;
                count += 1;
                int idx = indices[i];
                const vector<int>& encoder_token_ids = encoder_tokens_train[idx];
                const vector<int>& decoder_input_token_ids = decoder_input_token_ids_train[idx];
                const vector<int>& decoder_target_token_ids = decoder_target_token_ids_train[idx];
                const mx::array& forwarded = forward(encoder_token_ids, decoder_input_token_ids);
                backward(decoder_target_token_ids, forwarded);
            }
            normalize_gradients(end - start);
            optimizer->update_parameters();
            reset_gradients();
        }
        float loss = compute_validation_loss(encoder_tokens_val, decoder_tokens_val);
        cout << "Loss: " << loss << endl;
    }
}

void TransformerNetwork::save_model(const string& path) const {
    // We save:
    // 1. The number of encoder layers
    // 2. The number of decoder layers
    // 3. The input layer of the encoder
    // 4. Each encoder layer
    // 5. The input layer of the decoder
    // 6. Each decoder layer
    // 7. The final linear layer

    ofstream file(path);
    if (!file.is_open()) throw runtime_error("Could not open the file");
    file.good();
    file << setprecision(numeric_limits<float>::max_digits10);
    file << num_encoder_layers << " " << num_decoder_layers << " " << seq << " " << d_model << " " << h << " " << vocab_size << "\n";
    encoder_input_layer->save(file);
    for (Encoder* encoder: encoders) {
        encoder->save(file);
    }

    decoder_input_layer->save(file);
    for (Decoder* decoder: decoders) {
        decoder->save(file);
    }

    linear_layer->save(file);
    optimizer->save(file);
}

void TransformerNetwork::load_model(const string& path) {
    ifstream file(path);
    if (!file) throw runtime_error("Can not open file: " + path);

    file >> num_encoder_layers >> num_decoder_layers >> seq >> d_model >> h >> vocab_size;
    encoder_input_layer->load(file);
    for (Encoder* encoder: encoders) {
        encoder->load(file);
    }

    decoder_input_layer->load(file);
    for (Decoder* decoder: decoders) {
        decoder->load(file);
    }

    linear_layer->load(file);
    optimizer->load(file);
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
