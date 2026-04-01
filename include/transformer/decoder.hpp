#ifndef DECODER_HPP
#define DECODER_HPP

#include "transformer/layer_norm.hpp"
#include "transformer/multi_head_attention.hpp"
#include "transformer/feed_forward.hpp"

class Decoder {
    private:
        int seq, d_model, h, d_k, d_v, d_ff;
        LayerNorm* ln1;
        MultiHeadAttention* mha_masked;
        LayerNorm* ln2;
        MultiHeadAttention* mha_cross;
        LayerNorm* ln3;
        FeedForward* ff;
        vector<Layer*> layers;
        MatrixXd output;
        mlx::core::array output_mlx;
        MatrixXd d_input;
        mlx::core::array d_input_mlx;
        MatrixXd d_encoder_input;
        mlx::core::array d_encoder_input_mlx;
        ActivationFunction* activation;
        
    public:
        Decoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation);

        void forward(const MatrixXd& encoder_input, const MatrixXd& decoder_input);
        void forward_mlx(const mlx::core::array& encoder_input, const mlx::core::array& decoder_input);
        void backward(const MatrixXd& d_output);
        void backward_mlx(const mlx::core::array& d_output);
        MatrixXd infer(const MatrixXd& encoder_input, const MatrixXd& decoder_input);
        mlx::core::array infer_mlx(const mlx::core::array& encoder_input, const mlx::core::array& decoder_input);
        const MatrixXd& get_output() const;
        const mlx::core::array& get_output_mlx() const;
        const MatrixXd& get_d_input() const;
        const mlx::core::array& get_d_input_mlx() const;
        const MatrixXd& get_d_encoder_input() const;
        const mlx::core::array& get_d_encoder_input_mlx() const;
        const vector<Layer*>& get_layers();

        void save(ofstream& file) const;
        void load(ifstream& file);
};

#endif
