#ifndef BPE_TOKENIZER_HPP
#define BPE_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

using namespace std;

class BPETokenizer {
    private:
        using Word = vector<string>;

        struct WordEntry {
            Word tokens;
            int freq = 0;
        };

        // Learned merges in order.
        vector<pair<string, string>> merge_rules;

        unordered_map<string, int> token_to_id;
        vector<string> id_to_token;

    public:
        // Special tokens (fixed IDs)
        static constexpr const char* SOS = "<sos>";
        static constexpr const char* EOS = "<eos>";
        static constexpr const char* PAD = "<pad>";
        static constexpr const char* UNK = "<unk>";
        static constexpr int SOS_ID = 0;
        static constexpr int EOS_ID = 1;
        static constexpr int PAD_ID = 2;
        static constexpr int UNK_ID = 3;

        BPETokenizer();

        // Helpers
        static string to_lower_ascii(string sentence);
        static vector<string> split_whitespace(const string& sentence);

        static Word word_to_char_tokens(const string& word); // chars + </w>

        // Apply ONE merge to a token sequence
        static void apply_merge_inplace(Word& tokens, const string& a, const string& b);

        // Apply all learned merges to a word (used for encoding)
        void apply_all_merges_inplace(Word& tokens) const;

        // Build vocab from final corpus
        void build_vocab_from_entries(const vector<WordEntry>& entries, int vocab_size);

        // Count adjacent token pairs across corpus (weighted by word frequency)
        static void count_pair_frequencies(const vector<WordEntry>& entries, unordered_map<string, long long>& pair_freq);

        // Pair key encoding: "A\x1FB" (unit separator)
        static string make_token_pair(const string& a, const string& b);
        static void split_token_pair(const string& key, string& a, string& b);

        // Count unique tokens across all entries + 4 specials
        static int compute_token_count(const vector<WordEntry>& entries);

        // Train on a shared corpus (e.g., English + French sentences together).
        // vocab_size includes special tokens.
        void train(const vector<string>& sentences, int vocab_size);

        // Encode a sentence into its token IDs
        vector<int> encode(const string& raw_sentence, bool add_sos = false, bool add_eos = false) const;

        // Decode token IDs -> sentence (best-effort)
        string decode(const vector<int>& ids, bool remove_special = true) const;

        int get_vocab_size() const;
        int id_of(const string& token) const;

        const vector<string>& get_id_to_token() const;
        const unordered_map<string, int>& get_token_to_id() const;
        const vector<pair<string, string>>& get_merge_rules() const;

};

#endif