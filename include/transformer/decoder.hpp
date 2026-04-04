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
        std::vector<Layer*> layers;
        mlx::core::array output;
        mlx::core::array d_input;
        mlx::core::array d_encoder_input;
        ActivationFunction* activation;
        
    public:
        Decoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation);

        void forward(const mlx::core::array& encoder_input, const mlx::core::array& decoder_input);
        void backward(const mlx::core::array& d_output);
        mlx::core::array infer(const mlx::core::array& encoder_input, const mlx::core::array& decoder_input);
        const mlx::core::array& get_output() const;
        const mlx::core::array& get_d_input() const;
        const mlx::core::array& get_d_encoder_input() const;
        const std::vector<Layer*>& get_layers();

        void save(std::ofstream& file) const;
        void load(std::ifstream& file);
};

#endif
