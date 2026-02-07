#ifndef MULTI_HEAD_ATTENTION_HPP
#define MULTI_HEAD_ATTENTION_HPP

#include "core/layer.hpp"

enum class AttentionMode {
    ENCODER_SELF,
    DECODER_MASKED_SELF,
    DECODER_CROSS
};

class MultiHeadAttention : public Layer {
    private:
        AttentionMode mode;
        MultiHeadAttention* encoder_mha;
        VectorXd mean, inv_sqrt_var_plus_epsilon;
        RowVectorXd gamma_self, beta_self, d_gamma_self, d_beta_self;
        RowVectorXd gamma_cross, beta_cross, d_gamma_cross, d_beta_cross;
        MatrixXd E_hat, E_bar;
        MatrixXd diff;
        vector<MatrixXd> WQ, WK, WV, WO, d_WQ, d_WK, d_WV, d_WO;
        vector<MatrixXd> Q, K, V;
        vector<MatrixXd> softmaxJ, head;
        double momentum;
        int seq, d_model, h, d_k, d_v;
        int valid_len;

    public:
        MultiHeadAttention(int seq, int d_model, int h, AttentionMode mode, MultiHeadAttention* encoder = nullptr);

        void forward(const MatrixXd& input) override;

        void backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        const MatrixXd& get_output() const override;
        const MatrixXd& get_d_input() const override;

        string get_activation_name() const override;
        LayerType get_type() const override;

        AttentionMode get_mode() const;
        bool is_cross_attention() const;
        bool is_masked_attention() const;
        void set_valid_len(int valid_len);
};

#endif