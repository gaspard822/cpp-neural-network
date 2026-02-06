#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "transformer/layer_norm.hpp"
#include "transformer/multi_head_attention.hpp"
#include "transformer/feed_forward.hpp"

class Encoder {
    private:
        int seq, d_model, h, d_k, d_v;
        LayerNorm* norm_1;
        MultiHeadAttention* mha;
        LayerNorm* norm_2;
        FeedForward* feed_forward;
        
    public:
        Encoder(int seq, int d_model, int h, int d_k, int d_v);
};

#endif