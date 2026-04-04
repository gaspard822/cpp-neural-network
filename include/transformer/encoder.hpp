#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "transformer/layer_norm.hpp"
#include "transformer/multi_head_attention.hpp"
#include "transformer/feed_forward.hpp"

class Encoder {
    private:
        int seq, d_model, h, d_k, d_v, d_ff;
        LayerNorm* ln1;
        MultiHeadAttention* mha_self;
        LayerNorm* ln2;
        FeedForward* ff;
        std::vector<Layer*> layers;
        ActivationFunction* activation;
        mlx::core::array output;
        mlx::core::array d_input;
        
    public:
        Encoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation);
        void forward(const mlx::core::array& input);
        void backward(const mlx::core::array& d_output);
        mlx::core::array infer(const mlx::core::array& input);
        const mlx::core::array& get_output() const;
        const mlx::core::array& get_d_input() const;
        const std::vector<Layer*>& get_layers();

        void save(std::ofstream& file) const;
        void load(std::ifstream& file);
};

#endif
