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
        MatrixXd X;
        MatrixXd encoder_output, d_encoder_output;
        vector<MatrixXd> WQ, WK, WV, WO, d_WQ, d_WK, d_WV, d_WO;
        vector<MatrixXd> Q, K, V;
        vector<MatrixXd> softmaxJ, head;
        vector<TrainableParameter> params;
        int seq, d_model, h, d_k, d_v;

        // Pre-computed masks for masked attention
        MatrixXd forward_mask;
        MatrixXd backward_mask;

    public:
        MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode);

        void forward(const MatrixXd& input) override;

        void backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        void set_encoder_output(const MatrixXd& enc_out);

        const vector<TrainableParameter>& get_parameters() const override;
        const MatrixXd& get_output() const override;
        const MatrixXd& get_d_input() const override;
        const MatrixXd& get_d_encoder_output() const;

        string get_layer_name() const override;
        string get_activation_name() const override;
        LayerType get_type() const override;

        AttentionMode get_mode() const;
        bool is_cross_attention() const;
        bool is_masked_attention() const;

        void save(ofstream& file) const override;
        void load(ifstream& file) override;
};

#endif