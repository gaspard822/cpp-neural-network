#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include "core/network.hpp"
#include "transformer/input_layer.hpp"
#include "transformer/encoder.hpp"
#include "transformer/decoder.hpp"
#include "transformer/linear_layer.hpp"
#include "core/loss_function.hpp"
#include "core/optimizer.hpp"

class TransformerNetwork : public Network {
    private:
        int num_encoder_layers, num_decoder_layers;
        int seq, d_model, h;
        int d_k, d_v;  // We simply set d_k and d_v to d_model / h
        int d_ff;  // We simply set d_ff to 4 * d_model

        Tokenizer* tokenizer;
        ActivationFunction* activation;
        LossFunction* loss_function;
        Optimizer* optimizer;

        InputLayer* encoder_input_layer;
        vector<Encoder*> encoders;

        InputLayer* decoder_input_layer;
        vector<Decoder*> decoders;

        LinearLayer* linear_layer;
        
    public:
        TransformerNetwork(int num_encoder_layers, int num_decoder_layers, int seq, int d_model, int h,
                    Tokenizer* tokenizer, ActivationFunction* activation, LossFunction* loss_function, Optimizer* optimizer);

        ~TransformerNetwork();

        MatrixXd forward(const string& encoder_text, const string& decoder_text);
        void backward(const MatrixXd& y_true, const MatrixXd& y_pred);
        void train();

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
        Optimizer* get_optimizer() const override;

        // Straightforward getter
        NetworkType get_type() const override;
        
};

#endif