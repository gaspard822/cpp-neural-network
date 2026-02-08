#include <iostream>
#include "transformer/tokenizer.hpp"
#include "transformer/input_layer.hpp"
#include "transformer/multi_head_attention.hpp"
#include "mlp/neural_network.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "core/relu.hpp"
#include "transformer/transformer.hpp"


/*
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


pair<vector<VectorXi>, vector<VectorXi>> load_tokenized_sentences_from_csv(const string& csv_path, Tokenizer& tokenizer) {
    ifstream in(csv_path, ios::binary);
    if (!in) {
        throw runtime_error("Can not open CSV file: " + csv_path);
    }

    // Skip the header
    string header;
    getline(in, header);

    vector<VectorXi> en_sentences;
    vector<VectorXi> fr_sentences;

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
*/


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


void doing_stuff() {
    NeuralNetwork nn("CrossEntropy", "VanillaSGD");

    int seq = 2, d_model = 4, h = 2;
    int d_k, d_v = d_model / h;
    const string path_to_text = "../translation/en-fr-short.csv";

    Tokenizer* tokenizer = new Tokenizer(path_to_text, 100000);

    test_tokenizer(tokenizer);
}


void train_test_translation() {
    NeuralNetwork encoder(nullptr, "Adam");
    
}