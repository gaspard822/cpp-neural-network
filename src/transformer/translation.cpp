#include <iostream>
#include <fstream>
#include "transformer/transformer.hpp"
#include "core/relu.hpp"
#include "core/cross_entropy_loss.hpp"
#include "core/adam_optimizer.hpp"
#include "transformer/bpe_tokenizer.hpp"

struct DatasetSplit {
    vector<vector<int>> en_train;
    vector<vector<int>> fr_train;

    vector<vector<int>> en_val;
    vector<vector<int>> fr_val;

    vector<vector<int>> en_test;
    vector<vector<int>> fr_test;
};

void test_tokenizer(BPETokenizer* tokenizer) {
    const string sentence = "Hello, my name is Benoît";
    vector<int> encoding = tokenizer->encode(sentence);
    string decoding = tokenizer->decode(encoding);
    cout << "Sentence: " << sentence << endl;
    cout << "Encoding: ";
    for (auto it = encoding.begin(); it != encoding.end(); it++) {
        cout << *it << ", ";
    }
    cout << endl;
    cout << "Decoding: " << decoding << endl;
}

static inline pair<string, string> parse_csv_line_two_columns(const string& line) {
    string col1, col2;
    string* current = &col1;

    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            in_quotes = !in_quotes;
            continue; // do not include quotes themselves
        }

        if (c == ',' && !in_quotes) {
            if (current == &col2) {
                throw runtime_error("CSV line has more than two columns");
            }
            current = &col2;
            continue;
        }

        current->push_back(c);
    }

    if (current != &col2) {
        throw runtime_error("CSV line does not have two columns");
    }

    return {col1, col2};
}

pair<vector<vector<int>>, vector<vector<int>>> load_tokenized_sentences_from_csv(const string& csv_path, BPETokenizer& tokenizer, int max_num_sentences, int max_len_sentences) {
    ifstream in(csv_path, ios::binary);
    if (!in) throw runtime_error("Can not open CSV file: " + csv_path);

    // Skip the header
    string header;
    getline(in, header);

    vector<vector<int>> en_sentences;
    vector<vector<int>> fr_sentences;

    string line;
    int num_encoded_sentences = 0;
    while (getline(in, line) && (num_encoded_sentences < max_num_sentences)) {
        if (line.empty()) continue;

        auto [en_text, fr_text] = parse_csv_line_two_columns(line);

        vector<int> en_encoding = tokenizer.encode(en_text, true, true);
        vector<int> fr_encoding = tokenizer.encode(fr_text, true, true);
        if ((en_encoding.size() > max_len_sentences) || (fr_encoding.size() > max_len_sentences)) continue;
        if ((en_encoding.size() < 5) || (fr_encoding.size() < 5)) continue;
        en_sentences.push_back(en_encoding);
        fr_sentences.push_back(fr_encoding);

        num_encoded_sentences++;
        if (num_encoded_sentences % 100000 == 0) {
            cout << "Encoded " << num_encoded_sentences << " rows" << endl;
        }
    }

    if (en_sentences.size() != fr_sentences.size()) {
        throw runtime_error("English/French sentence count mismatch");
    }

    return {en_sentences, fr_sentences};
}

DatasetSplit split_dataset(const vector<vector<int>>& en_tokens, const vector<vector<int>>& fr_tokens, double train_size, double val_size) {
    int N = en_tokens.size();

    int num_train_samples = static_cast<int>(train_size * N);
    int num_val_samples = static_cast<int>(val_size * N);

    DatasetSplit split;

    split.en_train.assign(en_tokens.begin(), en_tokens.begin() + num_train_samples);
    split.fr_train.assign(fr_tokens.begin(), fr_tokens.begin() + num_train_samples);

    split.en_val.assign(en_tokens.begin() + num_train_samples, en_tokens.begin() + num_train_samples + num_val_samples);
    split.fr_val.assign(fr_tokens.begin() + num_train_samples, fr_tokens.begin() + num_train_samples + num_val_samples);

    split.en_test.assign(en_tokens.begin() + num_train_samples + num_val_samples, en_tokens.end());
    split.fr_test.assign(fr_tokens.begin() + num_train_samples + num_val_samples, fr_tokens.end());

    return split;
}

pair<vector<string>, vector<string>> load_sentences(const string& csv_path, int max_num_sentences, int max_len_sentences) {
    ifstream in(csv_path, ios::binary);
    if (!in) {
        throw runtime_error("Can not open CSV file: " + csv_path);
    }

    // Skip the header
    string header;
    getline(in, header);

    vector<string> en_sentences;
    vector<string> fr_sentences;

    string line;
    int num_encoded_sentences = 0;
    while (getline(in, line) && (num_encoded_sentences < max_num_sentences)) {
        if (line.empty()) continue;

        auto [en_text, fr_text] = parse_csv_line_two_columns(line);

        if ((en_text.size() > max_len_sentences) || (fr_text.size() > max_len_sentences)) continue;
        en_sentences.push_back(en_text);
        fr_sentences.push_back(fr_text);

        num_encoded_sentences++;
        if (num_encoded_sentences % 100000 == 0) {
            cout << "Encoded " << num_encoded_sentences << " rows" << endl;
        }
    }

    if (en_sentences.size() != fr_sentences.size()) {
        throw runtime_error("English/French sentence count mismatch");
    }

    return {en_sentences, fr_sentences};
}

void test_bpe_tokenizer() {
    vector<string> corpus;
    auto [en_sentences, fr_sentences] = load_sentences("../translation/en-fr-shuffled.csv", 10000, 128);
    corpus.reserve(en_sentences.size() + fr_sentences.size());
    for (auto& s : en_sentences) corpus.push_back(s);
    for (auto& s : fr_sentences) corpus.push_back(s);

    BPETokenizer tok;
    tok.train(corpus, 2000);

    auto ids = tok.encode("Hello world", true, true);
    cout << "Encoding: " << endl;
    for (auto id: ids) {
        cout << id << ", ";
    }
    cout << endl;
    auto text = tok.decode(ids);
    cout << "Decoding: " << text << endl;
}

void train_test_translation() {
    // test_bpe_tokenizer();
    // return;
    const string path_to_text = "../translation/en-fr-shuffled.csv";
    int num_encoder_layers = 3;
    int num_decoder_layers = 3;
    int seq = 128;
    int d_model = 256;
    int h = 4;
    int vocab_size = 10000;
    int num_epochs = 1;
    int batch_size = 1024;

    BPETokenizer* tokenizer = new BPETokenizer;
    vector<string> corpus;
    auto [en_sentences, fr_sentences] = load_sentences("../translation/en-fr-shuffled.csv", 10000, seq);
    corpus.reserve(en_sentences.size() + fr_sentences.size());
    for (auto& en_sentence : en_sentences) corpus.push_back(en_sentence);
    for (auto& fr_sentence: fr_sentences) corpus.push_back(fr_sentence);
    tokenizer->train(corpus, vocab_size);

    // test_tokenizer(tokenizer);
    ActivationFunction* activation = new Relu();
    Optimizer* optimizer = new AdamOptimizer(nullptr);

    int N = 100000;
    double train_size = 0.8;
    double val_size = 0.1;
    auto [en_tokens, fr_tokens] = load_tokenized_sentences_from_csv(path_to_text, *tokenizer, N, seq);
    DatasetSplit data = split_dataset(en_tokens, fr_tokens, train_size, val_size);

    TransformerNetwork* transformer_network = new TransformerNetwork(num_encoder_layers, num_decoder_layers, seq, d_model, h, tokenizer->get_vocab_size(), activation, optimizer);
    transformer_network->get_optimizer()->update_optimizer();
    chrono::time_point<chrono::high_resolution_clock> start, end;
    start = chrono::high_resolution_clock::now();
    transformer_network->train(data.en_train, data.fr_train, data.en_val, data.fr_val, num_epochs, batch_size);
    end = chrono::high_resolution_clock::now();
    cout << "N=" << N << " ; num_epochs=" << num_epochs << " ; batch_size=" << batch_size << " ; vocab_size=" << vocab_size << ": Took " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    transformer_network->save_model("../transformer_models/test_long_training.txt");
    
    transformer_network->infer(data.en_test, tokenizer, "../translation/output.csv");
    // TransformerNetwork* loaded_transformer_network = new TransformerNetwork(num_encoder_layers, num_decoder_layers, seq, d_model, h, tokenizer->get_vocab_size(), activation, cross_entropy_loss, optimizer);
    // cout << "Loading the model" << endl;
    // transformer_network->load_model("../tansformer_models/test_long_training_16_09.txt");
    // vector<int> sentence_tokens = tokenizer->encode("in january 1996, government accepted claim for negotiation while inquiry underway.");
    // vector<int> decoder_tokens = {0};
    // cout << transformer_network->forward(sentence_tokens, decoder_tokens);
}
