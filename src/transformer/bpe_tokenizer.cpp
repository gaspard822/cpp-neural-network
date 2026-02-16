#include "transformer/bpe_tokenizer.hpp"

#include <fstream>
#include <stdexcept>
#include <unordered_set>

using namespace std;

BPETokenizer::BPETokenizer() {
    id_to_token.push_back(SOS); token_to_id[SOS] = SOS_ID;
    id_to_token.push_back(EOS); token_to_id[EOS] = EOS_ID;
    id_to_token.push_back(PAD); token_to_id[PAD] = PAD_ID;
    id_to_token.push_back(UNK); token_to_id[UNK] = UNK_ID;
}

int BPETokenizer::get_vocab_size() const {
    return id_to_token.size();
}

int BPETokenizer::id_of(const string& token) const {
    auto token_it = token_to_id.find(token);
    if (token_it == token_to_id.end()) return token_to_id.at(UNK);
    return token_it->second;
}

string BPETokenizer::to_lower_ascii(string sentence) {
    for (char& c: sentence) c = (char)tolower((unsigned char)c);
    return sentence;
}

vector<string> BPETokenizer::split_whitespace(const string& sentence) {
    vector<string> words;
    string current_word;
    for (char c: sentence) {
        if (isspace((unsigned char)c)) {
            if (!current_word.empty()) { words.push_back(current_word); current_word.clear(); }
        } else {
            current_word.push_back(c);
        }
    }
    if (!current_word.empty()) words.push_back(current_word);
    return words;
}

BPETokenizer::Word BPETokenizer::word_to_char_tokens(const string& word) {
    Word tokens;
    tokens.reserve(word.size() + 1);
    for (char c: word) tokens.push_back(string(1, c));
    tokens.push_back("</w>");
    return tokens;
}

void BPETokenizer::apply_merge_inplace(Word& tokens, const string& a, const string& b) {
    if (tokens.size() < 2) return;

    Word new_tokens;
    new_tokens.reserve(tokens.size());

    for (int i = 0; i < tokens.size(); i++) {
        if (i + 1 < tokens.size() && tokens[i] == a && tokens[i + 1] == b) {
            new_tokens.push_back(a + b);
            i += 1;
        } else {
            new_tokens.push_back(tokens[i]);
        }
    }
    tokens.swap(new_tokens);
}

void BPETokenizer::apply_all_merges_inplace(Word& tokens) const {
    for (const auto& merge_rule: merge_rules) {
        apply_merge_inplace(tokens, merge_rule.first, merge_rule.second);
    }
}

string BPETokenizer::make_token_pair(const string& a, const string& b) {
    // Use a separator unlikely to appear in normal text (ASCII Unit Separator (0x1F))
    string token_pair;
    token_pair.reserve(a.size() + 1 + b.size());
    token_pair += a;
    token_pair.push_back('\x1F');
    token_pair += b;
    return token_pair;
}

void BPETokenizer::split_token_pair(const string& token_pair, string& a, string& b) {
    int pos = token_pair.find('\x1F');
    a = token_pair.substr(0, pos);
    b = token_pair.substr(pos + 1);
}

void BPETokenizer::count_pair_frequencies(const vector<WordEntry>& entries, unordered_map<string, long long>& pair_freq) {
    pair_freq.clear();

    for (const auto& entry: entries) {
        const Word& word = entry.tokens;
        if (word.size() <= 1) continue;

        for (size_t i = 0; i + 1 < word.size(); i++) {
            pair_freq[make_token_pair(word[i], word[i + 1])] += (long long)entry.freq;
        }
    }
}

void BPETokenizer::build_vocab_from_entries(const vector<WordEntry>& entries, int target_vocab_size) {
    // Collect all tokens that appear in the final corpus
    unordered_set<string> token_set;
    for (const auto& entry: entries) {
        for (const auto& token: entry.tokens) {
            token_set.insert(token);
        }
    }

    // Reset vocab and add special tokens
    token_to_id.clear();
    id_to_token.clear();
    id_to_token.push_back(SOS); token_to_id[SOS] = SOS_ID;
    id_to_token.push_back(EOS); token_to_id[EOS] = EOS_ID;
    id_to_token.push_back(PAD); token_to_id[PAD] = PAD_ID;
    id_to_token.push_back(UNK); token_to_id[UNK] = UNK_ID;

    // Add all corpus tokens. If token_set is larger than target, we still add all
    // because BPE training stops based on token set size; token_set should be <= target.
    vector<string> tokens(token_set.begin(), token_set.end());
    sort(tokens.begin(), tokens.end()); // deterministic

    for (const auto& token: tokens) {
        if (id_to_token.size() >= target_vocab_size) break;
        if (token_to_id.find(token) != token_to_id.end()) continue; // skip specials if collided (unlikely)
        token_to_id[token] = id_to_token.size();
        id_to_token.push_back(token);
    }
}

int BPETokenizer::compute_token_count(const vector<WordEntry>& entries) {
    unordered_set<string> token_set;
    for (const auto& entry: entries) {
        for (const auto& token: entry.tokens) {
            token_set.insert(token);
        }
    }
    return token_set.size() + 4;
}

void BPETokenizer::train(const vector<string>& sentences, int target_vocab_size) {

    // 1. Compute the frequency of each word
    unordered_map<string, int> word_freq;
    word_freq.reserve(sentences.size() * 4);

    for (const auto& sentence_raw : sentences) {
        string sentence = to_lower_ascii(sentence_raw);
        auto words = split_whitespace(sentence);
        for (auto& word : words) {
            if (!word.empty()) word_freq[word] += 1;
        }
    }

    // 2. Map each unique word to (tokenized chars + </w>) and its freq
    vector<WordEntry> entries;
    entries.reserve(word_freq.size());
    for (const auto& kv : word_freq) {
        WordEntry entry;
        entry.tokens = word_to_char_tokens(kv.first);
        entry.freq = kv.second;
        entries.push_back(std::move(entry));
    }

    // 3. Repeatedly merge most frequent pair until vocab size reached or no pairs
    unordered_map<string, long long> pair_freq;
    while (true) {
        int current_vocab_size = compute_token_count(entries);
        if (current_vocab_size >= target_vocab_size) break;

        count_pair_frequencies(entries, pair_freq);

        // Find best pair (max frequency)
        if (pair_freq.empty()) break;
        auto best_pair_it = pair_freq.begin();
        for (auto current_pair_it = pair_freq.begin(); current_pair_it != pair_freq.end(); current_pair_it++) {
            if (current_pair_it->second > best_pair_it->second) best_pair_it = current_pair_it;
        }

        // If best frequency is tiny, merging may not help; still proceed unless 0.
        if (best_pair_it->second <= 0) break;

        string a, b;
        split_token_pair(best_pair_it->first, a, b);

        // Record merge
        merge_rules.push_back({a, b});

        // Apply merge to all words
        for (auto& entry: entries) {
            apply_merge_inplace(entry.tokens, a, b);
        }
    }

    // 4. Build final vocab
    build_vocab_from_entries(entries, target_vocab_size);
}

vector<int> BPETokenizer::encode(const string& raw_sentence, bool add_sos, bool add_eos) const {
    vector<int> token_ids;
    if (add_sos) token_ids.push_back(token_to_id.at(SOS));

    string sentence = to_lower_ascii(raw_sentence);
    auto words = split_whitespace(sentence);

    // Encode each word into subword tokens, then emit ids.
    // We also insert a space between words by relying on </w> markers in decoding.
    for (auto& word: words) {
        if (word.empty()) continue;

        Word tokens = word_to_char_tokens(word);
        apply_all_merges_inplace(tokens);

        for (const auto& token: tokens) {
            auto token_it = token_to_id.find(token);
            token_ids.push_back(token_it == token_to_id.end() ? token_to_id.at(UNK) : token_it->second);
        }
    }

    if (add_eos) token_ids.push_back(token_to_id.at(EOS));
    return token_ids;
}

string BPETokenizer::decode(const vector<int>& token_ids, bool remove_special) const {
    string sentence;
    sentence.reserve(token_ids.size() * 4);

    for (int token_id: token_ids) {
        if (token_id < 0 || token_id >= id_to_token.size()) continue;

        const string& token = id_to_token[token_id];
        if (remove_special) {
            if (token == PAD || token == SOS || token == EOS) continue;
        }

        sentence += token;
    }

    // Replace "</w>" with space
    const string marker = "</w>";
    int position = 0;
    while ((position = sentence.find(marker, position)) != string::npos) {
        sentence.replace(position, marker.size(), " ");
        position += 1;
    }

    // Trim trailing spaces
    while (!sentence.empty() && isspace((unsigned char)sentence.back())) sentence.pop_back();

    return sentence;
}


const vector<string>& BPETokenizer::get_id_to_token() const {
    return id_to_token;
}

const unordered_map<string, int>& BPETokenizer::get_token_to_id() const {
    return token_to_id;
}

const vector<pair<string, string>>& BPETokenizer::get_merge_rules() const {
    return merge_rules;
}

void BPETokenizer::save(const string& path) const {
    ofstream file(path);
    if (!file.is_open()) throw runtime_error("Could not open file for writing: " + path);

    // Save vocabulary size
    file << id_to_token.size() << "\n";

    // Save all tokens with length-prefixed format (to handle spaces and special chars)
    for (const string& token : id_to_token) {
        file << token.length() << " " << token << "\n";
    }

    // Save merge rules
    file << merge_rules.size() << "\n";
    for (const auto& [first, second] : merge_rules) {
        file << first.length() << " " << first << " ";
        file << second.length() << " " << second << "\n";
    }

    file.close();
}

void BPETokenizer::load(const string& path) {
    ifstream file(path);
    if (!file.is_open()) throw runtime_error("Could not open file for reading: " + path);

    // Clear existing data
    id_to_token.clear();
    token_to_id.clear();
    merge_rules.clear();

    // Load vocabulary
    int vocab_size;
    file >> vocab_size;

    id_to_token.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        int length;
        file >> length;
        file.ignore(1); // skip space

        string token;
        token.resize(length);
        file.read(&token[0], length);
        file.ignore(1); // skip newline

        id_to_token.push_back(token);
        token_to_id[token] = i;
    }

    // Load merge rules
    int num_merges;
    file >> num_merges;

    merge_rules.reserve(num_merges);
    for (int i = 0; i < num_merges; i++) {
        int len1, len2;
        file >> len1;
        file.ignore(1); // skip space

        string first, second;
        first.resize(len1);
        file.read(&first[0], len1);

        file >> len2;
        file.ignore(1); // skip space

        second.resize(len2);
        file.read(&second[0], len2);
        file.ignore(1); // skip newline

        merge_rules.push_back({first, second});
    }

    file.close();
}
