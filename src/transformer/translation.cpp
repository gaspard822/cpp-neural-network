#include <iostream>

#include "transformer/transformer.hpp"
#include "core/relu.hpp"
#include "core/cross_entropy_loss.hpp"
#include "core/adam_optimizer.hpp"
#include "transformer/tokenizer.hpp"

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


pair<vector<vector<int>>, vector<vector<int>>> load_tokenized_sentences_from_csv(const string& csv_path, Tokenizer& tokenizer) {
    ifstream in(csv_path, ios::binary);
    if (!in) {
        throw runtime_error("Can not open CSV file: " + csv_path);
    }

    // Skip the header
    string header;
    getline(in, header);

    vector<vector<int>> en_sentences;
    vector<vector<int>> fr_sentences;

    string line;
    int i = 0;
    while (getline(in, line)) {
        if (i % 1000000 == 0) {
            cout << "Encoded " << i << " rows" << endl;
        }
        i++;
        if (line.empty()) continue;

        auto [en_text, fr_text] = parse_csv_line_two_columns(line);

        en_sentences.push_back(tokenizer.encode(en_text));
        fr_sentences.push_back(tokenizer.encode(fr_text));
    }

    if (en_sentences.size() != fr_sentences.size()) {
        throw runtime_error("English/French sentence count mismatch");
    }

    return {en_sentences, fr_sentences};
}

void test_tokenizer(Tokenizer* tokenizer) {
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

void train_test_translation() {
    const string path_to_text = "../translation/en-fr-short.csv";
    int num_encoder_layers = 2;
    int num_decoder_layers = 2;
    int seq = 256;
    int d_model = 64;
    int h = 4;
    Tokenizer* tokenizer = new Tokenizer(path_to_text, 100000);
    // test_tokenizer(tokenizer);
    ActivationFunction* activation = new Relu();
    CrossEntropy* cross_entropy_loss = new CrossEntropy();
    Optimizer* optimizer = new AdamOptimizer(nullptr);

    TransformerNetwork* transformer_network = new TransformerNetwork(num_encoder_layers, num_decoder_layers, seq, d_model, h, tokenizer->get_vocab_size(), activation, cross_entropy_loss, optimizer);
    transformer_network->get_optimizer()->update_optimizer();

    auto [en_sentences, fr_sentences] = load_tokenized_sentences_from_csv(path_to_text, *tokenizer);
    transformer_network->train(en_sentences, fr_sentences, 100);
    
    const string encoder_sentence = "About two thirds of Canadians can no longer see the Milky Way.";
    vector<int> encoder_token_ids = tokenizer->encode(encoder_sentence);
    vector<int> predicted_token_ids = transformer_network->infer(encoder_token_ids);
    for (auto it = predicted_token_ids.begin(); it != predicted_token_ids.end(); it++) {
        cout << *it << ", ";
    }
    cout << endl;
    string predicted_sentence = tokenizer->decode(predicted_token_ids);
    cout << "Predicted: " << predicted_sentence << endl;
}
