#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class Tokenizer {
    public:
        static constexpr int PAD_ID = 0;
        static constexpr int SOS_ID = 1;
        static constexpr int EOS_ID = 2;
        static constexpr int UNK_ID = 3;

        Tokenizer(const string& csv_path, int max_rows = 1000);

        // Encode string to token IDs
        VectorXi encode(const string& text) const;

        // Decode token IDs back to string
        string decode(const VectorXi& ids) const;

    private:
        void build_vocab_from_csv();

        string csv_path;
        int max_rows;
        int vocab_size;

        unordered_map<unsigned char, int> byte_to_id;
        unordered_map<int, unsigned char> id_to_byte;
};

#endif