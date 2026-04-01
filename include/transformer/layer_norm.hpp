#ifndef LAYER_NORM_HPP
#define LAYER_NORM_HPP

#include "core/layer.hpp"


class LayerNorm: public Layer {
    private:
        VectorXd mean, inv_sqrt_var_plus_epsilon;
        mlx::core::array inv_sqrt_var_plus_epsilon_mlx;
        RowVectorXd gamma, beta, d_gamma, d_beta;
        mlx::core::array gamma_mlx, beta_mlx, d_gamma_mlx, d_beta_mlx;
        MatrixXd diff, normalized_input;
        mlx::core::array diff_mlx, normalized_input_mlx;
        vector<TrainableParameter> params;
        double epsilon;
        int seq, d_model;

    public:
        LayerNorm(int seq, int d_model);

        void forward(const MatrixXd& input) override;
        void forward_mlx(const mlx::core::array& input);

        void backward(const MatrixXd& d_output) override;
        void backward_mlx(const mlx::core::array& d_output);

        MatrixXd infer(const MatrixXd& layer_input) const override;
        mlx::core::array infer_mlx(const mlx::core::array& layer_input) const;

        const vector<TrainableParameter>& get_parameters() const override;
        const MatrixXd& get_output() const override;
        const mlx::core::array& get_output_mlx() const;
        const MatrixXd& get_d_input() const override;
        const mlx::core::array& get_d_input_mlx() const;

        string get_layer_name() const override;
        string get_activation_name() const override;
        LayerType get_type() const override;

        void save(ofstream& file) const override;
        void load(ifstream& file) override;
};

#endif