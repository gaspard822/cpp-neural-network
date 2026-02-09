#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "core/layer.hpp"
#include "transformer/tokenizer.hpp"

class InputLayer: Layer {
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

        void forward(const MatrixXd& input) override;
        void forward(const string& text);

        void backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        const MatrixXd& get_output() const override;
        const MatrixXd& get_d_input() const override;

        string get_activation_name() const override;
        LayerType get_type() const override;
};

#endif