#ifndef BPE_TOKENIZER_HPP
#define BPE_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

class BPETokenizer {
    private:
        using Word = std::vector<std::string>;

        struct WordEntry {
            Word tokens;
            int freq = 0;
        };

        // Learned merges in order.
        std::vector<std::pair<std::string, std::string>> merge_rules;

        std::unordered_map<std::string, int> token_to_id;
        std::vector<std::string> id_to_token;

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
        static std::string to_lower_ascii(std::string sentence);
        static std::vector<std::string> split_whitespace(const std::string& sentence);

        static Word word_to_char_tokens(const std::string& word); // chars + </w>

        // Apply ONE merge to a token sequence
        static void apply_merge_inplace(Word& tokens, const std::string& a, const std::string& b);

        // Apply all learned merges to a word (used for encoding)
        void apply_all_merges_inplace(Word& tokens) const;

        // Build vocab from final corpus
        void build_vocab_from_entries(const std::vector<WordEntry>& entries, int vocab_size);

        // Count adjacent token pairs across corpus (weighted by word frequency)
        static void count_pair_frequencies(const std::vector<WordEntry>& entries, std::unordered_map<std::string, long long>& pair_freq);

        // Pair key encoding: "A\x1FB" (unit separator)
        static std::string make_token_pair(const std::string& a, const std::string& b);
        static void split_token_pair(const std::string& key, std::string& a, std::string& b);

        // Count unique tokens across all entries + 4 specials
        static int compute_token_count(const std::vector<WordEntry>& entries);

        // Train on a shared corpus (e.g., English + French sentences together).
        // vocab_size includes special tokens.
        void train(const std::vector<std::string>& sentences, int vocab_size);

        // Encode a sentence into its token IDs
        std::vector<int> encode(const std::string& raw_sentence, bool add_sos = false, bool add_eos = false) const;

        // Decode token IDs -> sentence (best-effort)
        std::string decode(const std::vector<int>& ids, bool remove_special = true) const;

        int get_vocab_size() const;
        int id_of(const std::string& token) const;

        const std::vector<std::string>& get_id_to_token() const;
        const std::unordered_map<std::string, int>& get_token_to_id() const;
        const std::vector<std::pair<std::string, std::string>>& get_merge_rules() const;

        // Save tokenizer to file
        void save(const std::string& path) const;

        // Load tokenizer from file
        void load(const std::string& path);

};

#endif