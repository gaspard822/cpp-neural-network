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

vector<vector<int>> unpad_tokens(const mx::array& token_ids, int pad_id) {
    mx::eval(token_ids);
    int num_sentences = token_ids.shape(0);
    int sentence_length = token_ids.shape(1);
    const int* data = token_ids.data<int>();

    vector<vector<int>> result(num_sentences);
    for (int i = 0; i < num_sentences; i++) {
        for (int j = 0; j < sentence_length; j++) {
            int tok = data[i * sentence_length + j];
            if (tok == pad_id) break;
            result[i].push_back(tok);
        }
    }
    return result;
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

mx::array TransformerNetwork::infer(const mx::array& encoder_token_ids, const mx::array& encoder_padding_mask) const {
    int num_sentences = encoder_token_ids.shape(0);

    // Encode source sentences once
    mx::array encoder_output = encoder_input_layer->infer(encoder_token_ids);
    for (Encoder* encoder : encoders) {
        encoder_output = encoder->infer(encoder_output, encoder_padding_mask);
    }

    // Start decoder with SOS for each sentence: {num_sentences, 1}
    vector<int> init(num_sentences, BPETokenizer::SOS_ID);
    mx::array decoder_tokens = mx::array(init.data(), {num_sentences, 1}, mx::int32);

    vector<bool> done(num_sentences, false);
    int num_done = 0;

    while (num_done < num_sentences && decoder_tokens.shape(1) < seq) {
        int sentence_length = decoder_tokens.shape(1);
        mx::array dec_pad_mask = mx::zeros({num_sentences, 1, 1, sentence_length}, mx::float32);

        mx::array decoder_output = decoder_input_layer->infer(decoder_tokens);
        for (Decoder* decoder : decoders) {
            decoder_output = decoder->infer(encoder_output, decoder_output, encoder_padding_mask, dec_pad_mask);
        }
        mx::array logits = linear_layer->infer(decoder_output);  // {num_sentences, sentence_length, vocab_size}

        // Take last position's logits and argmax
        mx::array last_logits = mx::squeeze(mx::slice(logits, {0, sentence_length - 1, 0}, {num_sentences, sentence_length, logits.shape(2)}), 1);
        mx::array next_tokens = mx::argmax(last_logits, 1);
        mx::eval(next_tokens);
        const int* next_data = next_tokens.data<int>();

        vector<int> next_vec(num_sentences);
        for (int i = 0; i < num_sentences; i++) {
            if (done[i]) {
                next_vec[i] = BPETokenizer::PAD_ID;
            } else {
                next_vec[i] = next_data[i];
                if (next_data[i] == BPETokenizer::EOS_ID) {
                    done[i] = true;
                    num_done++;
                }
            }
        }
        mx::array next_col = mx::array(next_vec.data(), {num_sentences, 1}, mx::int32);
        decoder_tokens = mx::concatenate({decoder_tokens, next_col}, 1);
    }

    return decoder_tokens;  // {num_sentences, max_sentence_length}
}

vector<vector<int>> TransformerNetwork::infer(const vector<vector<int>>& encoder_tokens) const {
    int N = encoder_tokens.size();
    auto [encoder_token_ids, enc_lengths] = pad_and_measure(encoder_tokens, seq, BPETokenizer::PAD_ID);
    const mx::array mx_indices = mx::arange(N, mx::int32);

    vector<vector<int>> all_results;
    all_results.reserve(N);

    // We infer in batches of 1024 to not overload the memory
    for (int start = 0; start < N; start += 1024) {
        int end = min(start + 1024, N);
        int current_batch_size = end - start;
        const mx::array current_batch_indices = mx::slice(mx_indices, {start}, {end});

        // take only the sentences corresponding to the indices of the current batch
        mx::array enc_tokens = mx::take(encoder_token_ids, current_batch_indices, 0);

        // compute the max sentence lengths
        mx::array encoder_max_sentence_length = mx::max(mx::take(enc_lengths, current_batch_indices));
        int enc_max = encoder_max_sentence_length.item<int>();

        // cut off the tokens beyond the max sentence length
        enc_tokens = mx::slice(enc_tokens, {0, 0}, {current_batch_size, enc_max});

        // create padding masks with shape {current_batch_size, 1, 1, max_length}
        mx::array enc_pad = mx::equal(enc_tokens, mx::array(BPETokenizer::PAD_ID));
        mx::array encoder_padding_mask = mx::reshape(mx::where(enc_pad, mx::array(-1e15f), mx::array(0.0f)), {current_batch_size, 1, 1, enc_max});

        mx::array batch_result = infer(enc_tokens, encoder_padding_mask);
        vector<vector<int>> batch_tokens = unpad_tokens(batch_result, BPETokenizer::PAD_ID);
        all_results.insert(all_results.end(), batch_tokens.begin(), batch_tokens.end());
    }

    return all_results;
}

void TransformerNetwork::infer_live(BPETokenizer* tokenizer) const {
    cout << "Type a sentence and press Enter. Ctrl+C to quit." << endl;
    string line;
    while (true) {
        cout << "\n> ";
        if (!getline(cin, line)) break;
        if (line.empty()) continue;

        vector<int> tokens = tokenizer->encode(line, true, true);

        // Wrap as batch of 1
        int len = min((int)tokens.size(), seq);
        mx::array enc_tokens = mx::array(tokens.data(), {1, len}, mx::int32);

        mx::array enc_pad = mx::equal(enc_tokens, mx::array(BPETokenizer::PAD_ID));
        mx::array encoder_padding_mask = mx::reshape(mx::where(enc_pad, mx::array(-1e15f), mx::array(0.0f)), {1, 1, 1, len});

        mx::array result = infer(enc_tokens, encoder_padding_mask);  // {1, output_length}

        // Extract the single sentence (strip SOS, stop at EOS)
        mx::eval(result);
        int output_len = result.shape(1);
        const int* data = result.data<int>();
        vector<int> output_tokens;
        for (int t = 0; t < output_len; t++) {
            int tok = data[t];
            if (tok == BPETokenizer::PAD_ID || tok == BPETokenizer::EOS_ID) break;
            if (tok != BPETokenizer::SOS_ID) output_tokens.push_back(tok);
        }

        cout << tokenizer->decode(output_tokens) << endl;
    }
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
        int current_batch_size = end - start;

        mx::array enc_batch = mx::slice(enc_tokens, {start, 0}, {end, enc_tokens.shape(1)});
        mx::array dec_input_batch = mx::slice(dec_input_tokens, {start, 0}, {end, dec_input_tokens.shape(1)});
        mx::array dec_target_batch = mx::slice(dec_target_tokens, {start, 0}, {end, dec_target_tokens.shape(1)});

        mx::array enc_pad = mx::reshape(
            mx::where(mx::equal(enc_batch, mx::array(BPETokenizer::PAD_ID)), mx::array(-1e15f), mx::array(0.0f)),
            {current_batch_size, 1, 1, enc_batch.shape(1)});
        mx::array dec_pad = mx::reshape(
            mx::where(mx::equal(dec_input_batch, mx::array(BPETokenizer::PAD_ID)), mx::array(-1e15f), mx::array(0.0f)),
            {current_batch_size, 1, 1, dec_input_batch.shape(1)});

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
            if (((int)(start / batch_size) % 50) == 0) cout << "batch " << (int)(start/batch_size) << endl;
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
