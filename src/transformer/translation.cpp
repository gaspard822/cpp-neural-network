#include <iostream>
#include <fstream>
#include "transformer/transformer.hpp"
#include "transformer/bpe_tokenizer.hpp"
#include "core/relu.hpp"
#include "core/cross_entropy_loss.hpp"
#include "core/adam_optimizer.hpp"

using namespace std;

struct TrainingConfig {
    // Model architecture
    int num_encoder_layers = 4;
    int num_decoder_layers = 4;
    int seq = 60;
    int d_model = 512;
    int h = 8;
    int vocab_size = 16000;

    // Training hyperparameters
    int num_epochs = 1;
    int batch_size = 256;
    float learning_rate = 0.0001;
    float beta1 = 0.9;
    float beta2 = 0.999;

    // Dataset parameters
    int N = 400000;  // Number of sentence pairs to use
    float train_size = 0.95;
    float val_size = 0.05;

    // Paths - parallel text files
    string en_data_path = "../translation/opus.en-fr-train.en.txt";
    string fr_data_path = "../translation/opus.en-fr-train.fr.txt";
    string tokenizer_path = "../transformer_models/bpe_tokenizer.txt";
    string model_path = "../transformer_models/saved_model.txt";
    string output_path = "../translation/output.csv";
    string tokenized_cache_path = "../transformer_models/tokenized_cache.bin";
};

struct DatasetSplit {
    vector<vector<int>> en_train;
    vector<vector<int>> fr_train;

    vector<vector<int>> en_val;
    vector<vector<int>> fr_val;

    vector<vector<int>> en_test;
    vector<vector<int>> fr_test;
};

pair<vector<string>, vector<string>> load_sentences_from_parallel_files(
    const string& en_path, const string& fr_path,
    int max_num_sentences, int max_len_sentences) {

    ifstream en_file(en_path);
    ifstream fr_file(fr_path);

    if (!en_file) throw runtime_error("Cannot open English file: " + en_path);
    if (!fr_file) throw runtime_error("Cannot open French file: " + fr_path);

    vector<string> en_sentences;
    vector<string> fr_sentences;

    string en_line, fr_line;
    int count = 0;

    while (getline(en_file, en_line) && getline(fr_file, fr_line) && count < max_num_sentences) {
        // Skip empty lines
        if (en_line.empty() || fr_line.empty()) continue;

        // Filter by length
        if (en_line.size() > max_len_sentences || fr_line.size() > max_len_sentences) continue;

        en_sentences.push_back(en_line);
        fr_sentences.push_back(fr_line);
        count++;

        if (count % 10000 == 0) {
            cout << "Loaded " << count << " sentence pairs" << endl;
        }
    }

    if (en_sentences.size() != fr_sentences.size()) {
        throw runtime_error("English/French sentence count mismatch");
    }

    cout << "Total loaded: " << en_sentences.size() << " sentence pairs" << endl;

    return {en_sentences, fr_sentences};
}

void save_tokenized_data(const string& cache_path, const vector<vector<int>>& en_tokens, const vector<vector<int>>& fr_tokens) {
    ofstream file(cache_path, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open cache file for writing: " + cache_path);

    // Save number of sentence pairs
    int num_pairs = en_tokens.size();
    file.write(reinterpret_cast<const char*>(&num_pairs), sizeof(num_pairs));

    // Save each sentence pair
    for (int i = 0; i < num_pairs; i++) {
        // Save English sentence
        int en_len = en_tokens[i].size();
        file.write(reinterpret_cast<const char*>(&en_len), sizeof(en_len));
        file.write(reinterpret_cast<const char*>(en_tokens[i].data()), en_len * sizeof(int));

        // Save French sentence
        int fr_len = fr_tokens[i].size();
        file.write(reinterpret_cast<const char*>(&fr_len), sizeof(fr_len));
        file.write(reinterpret_cast<const char*>(fr_tokens[i].data()), fr_len * sizeof(int));
    }

    file.close();
    cout << "Saved " << num_pairs << " tokenized sentence pairs to " << cache_path << endl;
}

pair<vector<vector<int>>, vector<vector<int>>> load_tokenized_data(const string& cache_path) {
    ifstream file(cache_path, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open cache file for reading: " + cache_path);

    vector<vector<int>> en_tokens;
    vector<vector<int>> fr_tokens;

    // Read number of sentence pairs
    int num_pairs;
    file.read(reinterpret_cast<char*>(&num_pairs), sizeof(num_pairs));

    en_tokens.reserve(num_pairs);
    fr_tokens.reserve(num_pairs);

    // Read each sentence pair
    for (int i = 0; i < num_pairs; i++) {
        // Read English sentence
        int en_len;
        file.read(reinterpret_cast<char*>(&en_len), sizeof(en_len));
        vector<int> en_sentence(en_len);
        file.read(reinterpret_cast<char*>(en_sentence.data()), en_len * sizeof(int));
        en_tokens.push_back(en_sentence);

        // Read French sentence
        int fr_len;
        file.read(reinterpret_cast<char*>(&fr_len), sizeof(fr_len));
        vector<int> fr_sentence(fr_len);
        file.read(reinterpret_cast<char*>(fr_sentence.data()), fr_len * sizeof(int));
        fr_tokens.push_back(fr_sentence);

        if ((i + 1) % 10000 == 0) {
            cout << "Loaded " << (i + 1) << " tokenized sentence pairs" << endl;
        }
    }

    file.close();
    cout << "Total loaded: " << en_tokens.size() << " tokenized sentence pairs from cache" << endl;

    return {en_tokens, fr_tokens};
}

pair<vector<vector<int>>, vector<vector<int>>> load_tokenized_sentences_from_parallel_files(
    const string& en_path, const string& fr_path,
    BPETokenizer& tokenizer, int max_num_sentences, int max_len_sentences) {

    // First, load raw sentences (use a generous character limit; token-level filtering happens below)
    auto [en_sentences, fr_sentences] = load_sentences_from_parallel_files(en_path, fr_path, max_num_sentences, 1000);

    // Then tokenize them
    vector<vector<int>> en_tokens;
    vector<vector<int>> fr_tokens;
    en_tokens.reserve(en_sentences.size());
    fr_tokens.reserve(fr_sentences.size());

    int skipped_too_long = 0;
    int skipped_too_short = 0;

    for (int i = 0; i < en_sentences.size(); i++) {
        vector<int> en_encoding = tokenizer.encode(en_sentences[i], true, true);
        vector<int> fr_encoding = tokenizer.encode(fr_sentences[i], true, true);

        // Skip if too long
        if (en_encoding.size() > max_len_sentences || fr_encoding.size() > max_len_sentences) {
            skipped_too_long++;
            continue;
        }

        // Skip if too short
        if (en_encoding.size() < 3 || fr_encoding.size() < 3) {
            skipped_too_short++;
            continue;
        }

        en_tokens.push_back(en_encoding);
        fr_tokens.push_back(fr_encoding);

        if (en_tokens.size() % 10000 == 0) {
            cout << "Tokenized " << en_tokens.size() << " sentence pairs" << endl;
        }
    }

    cout << "Total tokenized: " << en_tokens.size() << " sentence pairs" << endl;
    cout << "Skipped (too long): " << skipped_too_long << endl;
    cout << "Skipped (too short): " << skipped_too_short << endl;

    return {en_tokens, fr_tokens};
}

DatasetSplit split_dataset(const vector<vector<int>>& en_tokens, const vector<vector<int>>& fr_tokens, float train_size, float val_size) {
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

void test_tokenizer(BPETokenizer* tokenizer) {
    const string sentence = "Hello, my name is Benoît";
    vector<int> encoding = tokenizer->encode(sentence, true, true);
    string decoding = tokenizer->decode(encoding);
    cout << "Sentence: " << sentence << endl;
    cout << "Encoding: ";
    for (auto it = encoding.begin(); it != encoding.end(); it++) {
        cout << *it << ", ";
    }
    cout << endl;
    cout << "Decoding: " << decoding << endl;
}

void train_and_save_tokenizer(TrainingConfig& config) {
    BPETokenizer* tokenizer = new BPETokenizer;
    vector<string> corpus;
    auto [en_sentences, fr_sentences] = load_sentences_from_parallel_files(config.en_data_path, config.fr_data_path, config.N, numeric_limits<int>::max());
    corpus.reserve(en_sentences.size() + fr_sentences.size());
    for (auto& en_sentence : en_sentences) corpus.push_back(en_sentence);
    for (auto& fr_sentence: fr_sentences) corpus.push_back(fr_sentence);
    tokenizer->train(corpus, config.vocab_size);
    tokenizer->save(config.tokenizer_path);
}

void init_transformer_model(TrainingConfig& cfg) {
    // Load tokenizer
    BPETokenizer* tokenizer = new BPETokenizer;
    tokenizer->load(cfg.tokenizer_path);
    TransformerNetwork* transformer_network = new TransformerNetwork(
        cfg.num_encoder_layers, cfg.num_decoder_layers,
        cfg.seq, cfg.d_model, cfg.h,
        tokenizer->get_vocab_size(), new Relu(), new AdamOptimizer(nullptr, cfg.learning_rate, cfg.beta1, cfg.beta2)
    );
    transformer_network->get_optimizer()->update_optimizer();
    transformer_network->save_model(cfg.model_path);
}

void train_transformer_model(TrainingConfig& cfg, bool save_transformer_model=true) {
    // Load tokenizer
    BPETokenizer* tokenizer = new BPETokenizer;
    tokenizer->load(cfg.tokenizer_path);

    // Load model
    ActivationFunction* activation = new Relu();
    Optimizer* optimizer = new AdamOptimizer(nullptr, cfg.learning_rate, cfg.beta1, cfg.beta2);
    TransformerNetwork* transformer_network = new TransformerNetwork(cfg.model_path, activation, optimizer);
    transformer_network->get_optimizer()->update_optimizer();
    
    chrono::time_point<chrono::high_resolution_clock> start, end;
    // Load and split dataset
    start = chrono::high_resolution_clock::now();
    vector<vector<int>> en_tokens, fr_tokens;

    // Try to load from cache first
    ifstream cache_check(cfg.tokenized_cache_path);
    if (cache_check.good()) {
        cout << "Loading tokenized data from cache" << endl;
        tie(en_tokens, fr_tokens) = load_tokenized_data(cfg.tokenized_cache_path);
    } else {
        cout << "Cache not found. Tokenizing sentences" << endl;
        tie(en_tokens, fr_tokens) = load_tokenized_sentences_from_parallel_files(cfg.en_data_path, cfg.fr_data_path, *tokenizer, cfg.N, cfg.seq);
        save_tokenized_data(cfg.tokenized_cache_path, en_tokens, fr_tokens);
    }

    DatasetSplit data = split_dataset(en_tokens, fr_tokens, cfg.train_size, cfg.val_size);
    end = chrono::high_resolution_clock::now();
    cout << "Time for loading/tokenizing sentences: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    // Train model
    start = chrono::high_resolution_clock::now();
    transformer_network->train(data.en_train, data.fr_train, data.en_val, data.fr_val, cfg.num_epochs, cfg.batch_size);
    end = chrono::high_resolution_clock::now();

    cout << "N=" << cfg.N << " ; epochs=" << cfg.num_epochs
         << " ; batch_size=" << cfg.batch_size << " ; vocab_size=" << cfg.vocab_size
         << " ; d_model=" << cfg.d_model << " ; h=" << cfg.h << endl;
    cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

    if (save_transformer_model) transformer_network->save_model(cfg.model_path);
}

void translate_to_csv(TrainingConfig& cfg) {
    ofstream file(cfg.output_path);
    if (!file.is_open()) throw runtime_error("Could not open the file");

    // Load tokenizer
    BPETokenizer* tokenizer = new BPETokenizer;
    tokenizer->load(cfg.tokenizer_path);

    // Load model
    ActivationFunction* activation = new Relu();
    Optimizer* optimizer = new AdamOptimizer(nullptr, cfg.learning_rate, cfg.beta1, cfg.beta2);
    TransformerNetwork* transformer_network = new TransformerNetwork(cfg.model_path, activation, optimizer);
    transformer_network->get_optimizer()->update_optimizer();

    vector<vector<int>> en_tokens, fr_tokens;
    // Try to load from cache first
    ifstream cache_check(cfg.tokenized_cache_path);
    if (cache_check.good()) {
        cout << "Loading tokenized data from cache" << endl;
        tie(en_tokens, fr_tokens) = load_tokenized_data(cfg.tokenized_cache_path);
    } else {
        cout << "Cache not found. Tokenizing sentences" << endl;
        tie(en_tokens, fr_tokens) = load_tokenized_sentences_from_parallel_files(cfg.en_data_path, cfg.fr_data_path, *tokenizer, cfg.N, cfg.seq);
        save_tokenized_data(cfg.tokenized_cache_path, en_tokens, fr_tokens);
    }

    DatasetSplit data = split_dataset(en_tokens, fr_tokens, cfg.train_size, cfg.val_size);

    vector<vector<int>> translated_fr_test = transformer_network->infer(data.en_test);

    int num_sentences = data.en_test.size();
    for (int i = 0; i < num_sentences; i++) {
        string input_sentence = tokenizer->decode(data.en_test[i]);
        string predicted_sentence = tokenizer->decode(translated_fr_test[i]);
        file << "\"" << input_sentence << "\"," << "\"" << predicted_sentence << "\"\n";
    }
    file.close();

}

void infer_live_translation(TrainingConfig& cfg) {
    // Load tokenizer
    BPETokenizer* tokenizer = new BPETokenizer;
    tokenizer->load(cfg.tokenizer_path);
    // Load model
    ActivationFunction* activation = new Relu();
    Optimizer* optimizer = new AdamOptimizer(nullptr, cfg.learning_rate, cfg.beta1, cfg.beta2);
    TransformerNetwork* transformer_network = new TransformerNetwork(cfg.model_path, activation, optimizer);
    transformer_network->infer_live(tokenizer);
}
