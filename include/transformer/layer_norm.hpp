#ifndef LAYER_NORM_HPP
#define LAYER_NORM_HPP

#include "core/layer.hpp"


class LayerNorm: public Layer {
    private:
        mlx::core::array inv_sqrt_var_plus_epsilon;
        mlx::core::array gamma, beta, d_gamma, d_beta;
        mlx::core::array diff, normalized_input;
        vector<TrainableParameter> params;
        double epsilon;
        int seq, d_model;

    public:
        LayerNorm(int seq, int d_model);

        void forward(const mlx::core::array& input) override;

        void backward(const mlx::core::array& d_output) override;

        mlx::core::array infer(const mlx::core::array& layer_input) const override;

        const vector<TrainableParameter>& get_parameters() const override;
        const mlx::core::array& get_output() const override;
        const mlx::core::array& get_d_input() const override;

        string get_layer_name() const override;
        string get_activation_name() const override;
        LayerType get_type() const override;

        void save(ofstream& file) const override;
        void load(ifstream& file) override;
};

#endif
