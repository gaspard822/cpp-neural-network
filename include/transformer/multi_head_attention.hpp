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
        mlx::core::array X;
        mlx::core::array encoder_output, d_encoder_output, padding_mask;
        mlx::core::array WQ, WK, WV, WO, d_WQ, d_WK, d_WV, d_WO;
        mlx::core::array Q, K, V;
        mlx::core::array softmaxJ, head;
        std::vector<TrainableParameter> params;
        int seq, d_model, h, d_k, d_v;

        // Pre-computed masks for masked attention
        mlx::core::array forward_mask;
        mlx::core::array backward_mask;

    public:
        MultiHeadAttention(int seq, int d_model, int h, int d_k, int d_v, AttentionMode mode);

        void forward(const mlx::core::array& input) override;

        void backward(const mlx::core::array& d_output) override;

        mlx::core::array infer(const mlx::core::array& layer) const override;

        // Self-attention infer (encoder self or decoder masked self)
        mlx::core::array infer(const mlx::core::array& input, const mlx::core::array& padding_mask) const;

        // Cross-attention infer (decoder cross)
        mlx::core::array infer(const mlx::core::array& input, const mlx::core::array& encoder_out, const mlx::core::array& encoder_padding_mask) const;

        void set_encoder_output(const mlx::core::array& enc_out);
        void set_padding_mask(const mlx::core::array& pad_mask);

        const std::vector<TrainableParameter>& get_parameters() const override;
        const mlx::core::array& get_output() const override;
        const mlx::core::array& get_d_input() const override;
        const mlx::core::array& get_d_encoder_output() const;

        std::string get_layer_name() const override;
        std::string get_activation_name() const override;
        LayerType get_type() const override;

        AttentionMode get_mode() const;
        bool is_cross_attention() const;
        bool is_masked_attention() const;

        void save(std::ofstream& file) const override;
        void load(std::ifstream& file) override;
};

#endif
