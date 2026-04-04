#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "core/layer.hpp"


class LinearLayer: public Layer {
    private:
        mlx::core::array X;
        mlx::core::array W, d_W;
        mlx::core::array b, d_b;
        std::vector<TrainableParameter> params;
        int d_model, vocab_size;

    public:
        LinearLayer(int d_model, int vocab_size);

        void forward(const mlx::core::array& input) override;

        void backward(const mlx::core::array& d_output) override;

        mlx::core::array infer(const mlx::core::array& layer_input) const override;

        const std::vector<TrainableParameter>& get_parameters() const override;
        const mlx::core::array& get_output() const override;
        const mlx::core::array& get_d_input() const override;

        std::string get_layer_name() const override;
        std::string get_activation_name() const override;
        LayerType get_type() const override;

        void save(std::ofstream& file) const override;
        void load(std::ifstream& file) override;
};

#endif
