#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <unordered_map>
#include <vector>

using namespace std;


class Tokenizer {
    private:
        void build_vocab_from_csv();

        string csv_path;
        int max_rows;
        int vocab_size;

        unordered_map<unsigned char, int> byte_to_id;
        unordered_map<int, unsigned char> id_to_byte;
    
    public:
        static constexpr int SOS_ID = 0;
        static constexpr int EOS_ID = 1;
        static constexpr int PAD_ID = 2;
        static constexpr int UNK_ID = 3;
        
        Tokenizer(const string& csv_path, int max_rows = 1000);

        // Encode string to token IDs
        vector<int> encode(const string& text) const;

        // Decode token IDs back to string
        string decode(const vector<int>& ids) const;

        const int get_vocab_size() const;
};

#endif