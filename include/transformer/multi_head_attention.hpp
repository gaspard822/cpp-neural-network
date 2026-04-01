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
        mlx::core::array X_mlx;
        MatrixXd encoder_output, d_encoder_output;
        mlx::core::array encoder_output_mlx, d_encoder_output_mlx;
        vector<MatrixXd> WQ, WK, WV, WO, d_WQ, d_WK, d_WV, d_WO;
        vector<mlx::core::array> WQ_mlx, WK_mlx, WV_mlx, WO_mlx, d_WQ_mlx, d_WK_mlx, d_WV_mlx, d_WO_mlx;
        vector<MatrixXd> Q, K, V;
        vector<mlx::core::array> Q_mlx, K_mlx, V_mlx;
        vector<MatrixXd> softmaxJ, head;
        vector<mlx::core::array> softmaxJ_mlx, head_mlx;
        vector<TrainableParameter> params;
        int seq, d_model, h, d_k, d_v;

        // Pre-computed masks for masked attention
        MatrixXd forward_mask;
        mlx::core::array forward_mask_mlx;
        MatrixXd backward_mask;
        mlx::core::array backward_mask_mlx;

    public:
        MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode);

        void forward(const MatrixXd& input) override;
        void forward_mlx(const mlx::core::array& input);

        void backward(const MatrixXd& d_output) override;
        void backward_mlx(const mlx::core::array& d_output);

        MatrixXd infer(const MatrixXd& layer_input) const override;
        mlx::core::array infer_mlx(const mlx::core::array& layer_input) const;

        void set_encoder_output(const MatrixXd& enc_out);
        void set_encoder_output_mlx(const mlx::core::array& enc_out);

        const vector<TrainableParameter>& get_parameters() const override;
        const MatrixXd& get_output() const override;
        const mlx::core::array& get_output_mlx() const;
        const MatrixXd& get_d_input() const override;
        const mlx::core::array& get_d_input_mlx() const;
        const MatrixXd& get_d_encoder_output() const;
        const mlx::core::array& get_d_encoder_output_mlx() const;

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