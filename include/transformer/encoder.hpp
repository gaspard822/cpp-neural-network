#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "transformer/layer_norm.hpp"
#include "transformer/multi_head_attention.hpp"
#include "transformer/feed_forward.hpp"

class Encoder {
    private:
        int seq, d_model, h, d_k, d_v, d_ff;
        vector<Layer*> layers;
        MatrixXd output, d_input;
        ActivationFunction* activation;
        
    public:
        Encoder(int seq, int d_model, int h, int d_k, int d_v, int d_ff, ActivationFunction* activation);
        void forward(const MatrixXd& input);
        void backward(const MatrixXd& d_output);
        const MatrixXd& get_output() const;
        const MatrixXd& get_d_input() const;
};

#endif