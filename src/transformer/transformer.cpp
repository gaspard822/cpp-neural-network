#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include "transformer/transformer.hpp"
#include "transformer/bpe_tokenizer.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
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

pair<mx::array, mx::array> pad_and_measure(const vector<vector<int>>& token_ids, int seq, int pad_id) {
    int num_sentences = token_ids.size();
    vector<int> flat_padded(num_sentences * seq, pad_id);
    vector<int> lengths(num_sentences);
    for (int i = 0; i < num_sentences; i++) {
        int len = min((int)token_ids[i].size(), seq);
        copy(token_ids[i].begin(), token_ids[i].begin()+len, flat_padded.begin() + i * seq);
        lengths[i] = len;
    }
    return {mx::array(flat_padded.data(), {num_sentences, seq}, mx::int32), mx::array(lengths.data(), {num_sentences}, mx::int32)};
}

const mx::array& TransformerNetwork::forward(const mx::array& encoder_token_ids, const mx::array& decoder_token_ids,
                                             const mx::array& encoder_padding_mask, const mx::array& decoder_padding_mask) {

    encoder_input_layer->forward(encoder_token_ids);
    const mx::array* encoder_output = &encoder_input_layer->get_output();
    // Forward that embedding into the encoders
    for (Encoder* encoder: encoders) {
        encoder->forward(*encoder_output, encoder_padding_mask);
        encoder_output = &encoder->get_output();
    }
    
    decoder_input_layer->forward(decoder_token_ids);
    const mx::array* decoder_output = &decoder_input_layer->get_output();
    // Forward that embedding into the decoders
    for (Decoder* decoder: decoders) {
        decoder->forward(*encoder_output, *decoder_output, encoder_padding_mask, decoder_padding_mask);
        decoder_output = &decoder->get_output();
    }
    
    linear_layer->forward(*decoder_output);
    return linear_layer->get_output();
}

const mx::array& TransformerNetwork::backward(const mx::array& y_true, const mx::array& y_pred) {
    mx::array d_loss = cross_entropy_loss->derivative(y_true, y_pred, BPETokenizer::PAD_ID);
    const mx::array* decoder_d_input = &d_loss;
    linear_layer->backward(*decoder_d_input);
    decoder_d_input = &linear_layer->get_d_input();

    int num_sentences = encoders.back()->get_output().shape(0);
    int num_encoder_tokens = encoders.back()->get_output().shape(1);
    mx::array encoder_d_input_buf = mx::zeros({num_sentences, num_encoder_tokens, d_model}, mx::float32);
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
    /*
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
    */
}

void TransformerNetwork::infer_live(BPETokenizer* tokenizer) const {
    /*
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
    */
}

float TransformerNetwork::compute_validation_loss(vector<vector<int>>& encoder_tokens_val, vector<vector<int>>& decoder_tokens_val, int batch_size) {
    auto [dec_input_val, dec_target_val] = preprocess_decoder_token_ids(decoder_tokens_val);

    auto [enc_tokens, enc_lengths] = pad_and_measure(encoder_tokens_val, seq, BPETokenizer::PAD_ID);
    auto [dec_input_tokens, dec_lengths] = pad_and_measure(dec_input_val, seq, BPETokenizer::PAD_ID);
    auto [dec_target_tokens, dec_target_lengths] = pad_and_measure(dec_target_val, seq, BPETokenizer::PAD_ID);

    int N = enc_tokens.shape(0);
    float total_loss = 0.0f;
    int num_batches = 0;

    for (int start = 0; start < N; start += batch_size) {
        int end = min(start + batch_size, N);
        int bs = end - start;

        mx::array enc_batch = mx::slice(enc_tokens, {start, 0}, {end, enc_tokens.shape(1)});
        mx::array dec_input_batch = mx::slice(dec_input_tokens, {start, 0}, {end, dec_input_tokens.shape(1)});
        mx::array dec_target_batch = mx::slice(dec_target_tokens, {start, 0}, {end, dec_target_tokens.shape(1)});

        mx::array enc_pad = mx::reshape(
            mx::where(mx::equal(enc_batch, mx::array(BPETokenizer::PAD_ID)), mx::array(-1e15f), mx::array(0.0f)),
            {bs, 1, 1, enc_batch.shape(1)});
        mx::array dec_pad = mx::reshape(
            mx::where(mx::equal(dec_input_batch, mx::array(BPETokenizer::PAD_ID)), mx::array(-1e15f), mx::array(0.0f)),
            {bs, 1, 1, dec_input_batch.shape(1)});

        mx::array y_pred = forward(enc_batch, dec_input_batch, enc_pad, dec_pad);
        total_loss += cross_entropy_loss->compute(dec_target_batch, y_pred, BPETokenizer::PAD_ID);
        num_batches++;
    }

    return total_loss / num_batches;
}

void TransformerNetwork::train(
    vector<vector<int>>& encoder_tokens_train, vector<vector<int>>& decoder_tokens_train,
    vector<vector<int>>& encoder_tokens_val, vector<vector<int>>& decoder_tokens_val,
    int epochs, int batch_size) {

    auto [decoder_input_token_ids_train, decoder_target_token_ids_train] = preprocess_decoder_token_ids(decoder_tokens_train);

    // Store the token ids in a padded mx::array, get the length of each sentence in an mx::array
    auto [encoder_token_ids, enc_lengths] = pad_and_measure(encoder_tokens_train, seq, BPETokenizer::PAD_ID);
    auto [decoder_input_token_ids, dec_lengths] = pad_and_measure(decoder_input_token_ids_train, seq, BPETokenizer::PAD_ID);
    auto [decoder_target_token_ids, _] = pad_and_measure(decoder_target_token_ids_train, seq, BPETokenizer::PAD_ID);

    for (int epoch = 0; epoch < epochs; epoch++) {
        cout << "Epoch " << epoch << endl;
        int N = encoder_tokens_train.size();

        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);  // 0, 1, 2, ..., N-1
        random_device rd;
        mt19937 gen(rd());
        shuffle(indices.begin(), indices.end(), gen);

        // transform indices from a vector to an mx::array
        const mx::array mx_indices = mx::array(indices.data(), {N}, mx::int32);

        int count = 0;
        chrono::time_point<chrono::high_resolution_clock> start_timer, end_timer;
        start_timer = chrono::high_resolution_clock::now();
        for (int start = 0; start < N; start += batch_size) {
            int end = min(start + batch_size, N);
            int current_batch_size = end - start;
            const mx::array current_batch_indices = mx::slice(mx_indices, {start}, {end});

            // take only the sentences corresponding to the indices of the current batch
            mx::array enc_tokens = mx::take(encoder_token_ids, current_batch_indices, 0);
            mx::array dec_input_tokens = mx::take(decoder_input_token_ids, current_batch_indices, 0);
            mx::array dec_target_tokens = mx::take(decoder_target_token_ids, current_batch_indices, 0);

            // compute the max sentence lengths
            mx::array encoder_max_sentence_length = mx::max(mx::take(enc_lengths, current_batch_indices));
            mx::array decoder_max_sentence_length = mx::max(mx::take(dec_lengths, current_batch_indices));
            int enc_max = encoder_max_sentence_length.item<int>();
            int dec_max = decoder_max_sentence_length.item<int>();

            // cut off the tokens beyond the max sentence length
            enc_tokens = mx::slice(enc_tokens, {0, 0}, {current_batch_size, enc_max});
            dec_input_tokens = mx::slice(dec_input_tokens, {0, 0}, {current_batch_size, dec_max});
            dec_target_tokens = mx::slice(dec_target_tokens, {0, 0}, {current_batch_size, dec_max});

            // create padding masks with shape {current_batch_size, 1, 1, max_length}
            mx::array enc_pad = mx::equal(enc_tokens, mx::array(BPETokenizer::PAD_ID));
            mx::array encoder_padding_mask = mx::reshape(mx::where(enc_pad, mx::array(-1e15f), mx::array(0.0f)), {current_batch_size, 1, 1, enc_max});
            mx::array dec_pad = mx::equal(dec_input_tokens, mx::array(BPETokenizer::PAD_ID));
            mx::array decoder_padding_mask = mx::reshape(mx::where(dec_pad, mx::array(-1e15f), mx::array(0.0f)), {current_batch_size, 1, 1, dec_max});

            const mx::array& forwarded = forward(enc_tokens, dec_input_tokens, encoder_padding_mask, decoder_padding_mask);
            backward(dec_target_tokens, forwarded);
            optimizer->update_parameters();
        }
        end_timer = chrono::high_resolution_clock::now();
        cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(end_timer - start_timer).count() << "ms" << endl;
        float loss = compute_validation_loss(encoder_tokens_val, decoder_tokens_val, batch_size);
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
