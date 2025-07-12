#ifndef MULTI_HEAD_ATTENTION_LAYER_HPP
#define MULTI_HEAD_ATTENTION_LAYER_HPP

#include "layer.hpp"

class MultiHeadAttentionLayer : public Layer {
    private:
        VectorXd gamma, beta, d_gamma, d_beta, mean, inv_sqrt_var_plus_epsilon, running_mean, running_variance;
        MatrixXd E_bar, E_hat;
        vector<MatrixXd> WQ, WK, WV, WO, d_WQ, d_WK, d_WV, d_WO;
        // MatrixXd WO, d_WO;
        // vector<MatrixXd> Q, K, V, d_Q, d_K, d_V;
        // DON'T NEED TO STORE d_Q, d_K, d_V, AS THEY ARE NOT USED FOR THE BACKPROPAGATION
        vector<MatrixXd> Q, K, V;
        vector<MatrixXd> softmaxJ, head;
        double momentum;
        int seq, d_model, h, d_k, d_v;
        bool masked;

    public:
        MultiHeadAttentionLayer(int seq, int d_model, int h, bool masked = false);

        void forward(const MatrixXd& input) override;

        MatrixXd backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        string get_activation_name() const override;
        LayerType get_type() const override;
};

#endif