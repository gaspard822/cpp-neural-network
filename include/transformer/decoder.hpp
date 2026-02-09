#ifndef DECODER_HPP
#define DECODER_HPP

#include "transformer/layer_norm.hpp"
#include "transformer/multi_head_attention.hpp"
#include "transformer/feed_forward.hpp"

class Decoder {
    private:
        int seq, d_model, h, d_k, d_v, d_ff;
        MultiHeadAttention* mha_cross;
        vector<Layer*> layers;
        MatrixXd output;
        ActivationFunction* activation;
        
    public:
        Decoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation);

        void forward(const MatrixXd& encoder_input, const MatrixXd& decoder_input);

        const MatrixXd& get_output() const;
};

#endif