#ifndef DECODER_HPP
#define DECODER_HPP

#include "transformer/layer_norm.hpp"
#include "transformer/multi_head_attention.hpp"
#include "transformer/feed_forward.hpp"

class Decoder {
    private:
        int seq, d_model, h, d_k, d_v;
        LayerNorm* norm_1;
        MultiHeadAttention* mha_1;
        LayerNorm* norm_2;
        MultiHeadAttention* mha_2;
        LayerNorm* norm_3;
        FeedForward* feed_forward;
        
    public:
        Decoder(int seq, int d_model, int h, int d_k, int d_v);
};

#endif