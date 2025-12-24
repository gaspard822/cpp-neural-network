#ifndef MULTI_HEAD_ATTENTION_LAYER_HPP
#define MULTI_HEAD_ATTENTION_LAYER_HPP

#include "layer.hpp"

enum class AttentionMode {
    ENCODER_SELF,
    DECODER_MASKED_SELF,
    DECODER_CROSS
};

class MultiHeadAttentionLayer : public Layer {
    private:
        AttentionMode mode;
        MultiHeadAttentionLayer* encoder_mha;
        VectorXd mean, inv_sqrt_var_plus_epsilon, running_mean, running_variance;
        VectorXd gamma_self, beta_self, d_gamma_self, d_beta_self;
        VectorXd gamma_cross, beta_cross, d_gamma_cross, d_beta_cross;
        MatrixXd E_hat, E_bar;
        vector<MatrixXd> WQ, WK, WV, WO, d_WQ, d_WK, d_WV, d_WO;
        vector<MatrixXd> Q, K, V;
        vector<MatrixXd> softmaxJ, head;
        double momentum;
        int seq, d_model, h, d_k, d_v;

    public:
        MultiHeadAttentionLayer(int seq, int d_model, int h, AttentionMode mode, MultiHeadAttentionLayer* encoder = nullptr);

        void forward(const MatrixXd& input) override;

        MatrixXd backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        string get_activation_name() const override;
        LayerType get_type() const override;
        AttentionMode get_mode() const;
        bool is_cross_attention() const;
        bool is_masked_attention() const;
};

#endif