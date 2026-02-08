#include <fstream>
#include <stdexcept>
#include <array>
#include <algorithm>
#include "transformer/tokenizer.hpp"


Tokenizer::Tokenizer(const string& csv_path, int max_rows) : csv_path(csv_path), max_rows(max_rows), vocab_size(4) {
    build_vocab_from_csv();
}

void Tokenizer::build_vocab_from_csv() {
    ifstream in(csv_path, ios::binary);
    if (!in) {
        throw runtime_error("Can not open CSV file: " + csv_path);
    }

    array<bool, 256> seen;
    seen.fill(false);

    string line;
    int rows = 0;
    while (rows < max_rows && getline(in, line)) {
        for (unsigned char c: line) {
            seen[c] = true;
        }
        rows++;
    }

    int next_id = 4; // already have ids for SOS, EOS, PAD, UNK
    for (int c = 0; c < 256; c++) {
        if (seen[c]) {
            byte_to_id[static_cast<unsigned char>(c)] = next_id;
            id_to_byte[next_id] = static_cast<unsigned char>(c);
            next_id++;
        }
    }
    vocab_size = next_id;
}

vector<int> Tokenizer::encode(const string& text) const {
    int i = 0;
    vector<int> ids(text.size() + 2);
    ids[i++] = SOS_ID;
    
    for (unsigned char c: text) {
        auto it = byte_to_id.find(c);
        if (it == byte_to_id.end()) {
            ids[i] = UNK_ID;
        } else {
            ids[i] = it->second;
        }
        i++;
    }

    ids[i] = EOS_ID;
    
    return ids;
}

string Tokenizer::decode(const vector<int>& ids) const {
    string chars;
    int ids_size = ids.size();
    chars.reserve(ids_size);

    for (int i = 0; i < ids_size; i++) {
        if (ids[i] == SOS_ID || ids[i] == EOS_ID || ids[i] == PAD_ID) {
            continue;
        }

        if (ids[i] == UNK_ID) {
            chars.push_back('$');
            continue;
        }

        auto it = id_to_byte.find(ids[i]);
        if (it == id_to_byte.end()) throw runtime_error("Invalid token id in decode(): " + to_string(ids[i]));
        chars.push_back(static_cast<char>(it->second));
    }

    return chars;
}

const int Tokenizer::get_vocab_size() const {
    return vocab_size;
}