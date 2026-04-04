#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "core/network.hpp"
#include "transformer/bpe_tokenizer.hpp"
#include "transformer/input_layer.hpp"
#include "transformer/encoder.hpp"
#include "transformer/decoder.hpp"
#include "transformer/linear_layer.hpp"
#include "core/cross_entropy_loss.hpp"
#include "core/optimizer.hpp"

class TransformerNetwork : public Network {
    private:
        int num_encoder_layers, num_decoder_layers;
        int seq, d_model, h;
        int d_k, d_v;  // We simply set d_k and d_v to d_model / h
        int d_ff;  // We simply set d_ff to 4 * d_model
        int vocab_size;

        ActivationFunction* activation;
        CrossEntropy* cross_entropy_loss;
        Optimizer* optimizer;

        InputLayer* encoder_input_layer;
        vector<Encoder*> encoders;

        InputLayer* decoder_input_layer;
        vector<Decoder*> decoders;

        LinearLayer* linear_layer;

        vector<Layer*> layers;

    public:
        TransformerNetwork(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h, int vocab_size,
                    ActivationFunction* activation, Optimizer* optimizer);

        TransformerNetwork(const string& path, ActivationFunction* activation, Optimizer* optimizer);

        ~TransformerNetwork();

        // Helper that initializes layers (called by both constructors)
        void init_layers();

        const mlx::core::array& forward(const vector<int>& encoder_token_ids, const vector<int>& decoder_token_ids);
        const mlx::core::array& backward(const vector<int>& y_true, const mlx::core::array& y_pred);
        void infer(const vector<vector<int>>& encoder_token_ids, BPETokenizer* tokenizer, const string& csv_path) const;
        void infer_live(BPETokenizer* tokenizer) const;
        float compute_validation_loss(vector<vector<int>>& encoder_tokens_val, vector<vector<int>>& decoder_tokens_val);
        void reset_gradients();
        void normalize_gradients(int batch_size);
        void train(
            vector<vector<int>>& encoder_tokens_train, vector<vector<int>>& decoder_tokens_train,
            vector<vector<int>>& encoder_tokens_val, vector<vector<int>>& decoder_tokens_val,
            int epochs, int batch_size
        );

        /**
         * Saves the model's architecture and parameters to a .txt file.
         * @param path Filesystem path to save the model
         */
        void save_model(const string& path) const override;

        /**
         * Loads a model's architecture and parameters from a file.
         * @param path Filesystem path to the saved model
         */
        void load_model(const string& path) override;

        // Straightforward getter
        const vector<Layer*>& get_layers() const override;

        // Straightforward getter
        Optimizer* get_optimizer() const override;

        // Straightforward getter
        NetworkType get_type() const override;
        
};

#endif