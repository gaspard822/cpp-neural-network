#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "core/layer.hpp"
#include "transformer/tokenizer.hpp"

class InputLayer {
    private:
        int seq, d_model, vocab_size;
        Tokenizer* tokenizer;
        MatrixXd embeddings, d_embeddings;
        MatrixXd positional_encodings;
        
        vector<int> token_ids;
        MatrixXd output;

    public:
        InputLayer(int seq, int d_model, Tokenizer* tokenizer);

        MatrixXd compute_positional_encodings(int seq, int d_model);

        void forward(const string& text);

        void backward(const MatrixXd& d_output);

        MatrixXd infer(const MatrixXd& layer_input) const;

        unique_ptr<Gradients> get_gradients();
        unique_ptr<Gradients> get_params();
        const MatrixXd& get_output() const;

        string get_activation_name() const;
        LayerType get_type() const;
};

#endif