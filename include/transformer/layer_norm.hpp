#ifndef LAYER_NORM_HPP
#define LAYER_NORM_HPP

#include "core/layer.hpp"


class LayerNorm: public Layer {
    private:
        VectorXd mean, inv_sqrt_var_plus_epsilon;
        RowVectorXd gamma, beta, d_gamma, d_beta;
        MatrixXd diff, normalized_input;
        double epsilon;
        int seq, d_model;

    public:
        LayerNorm(int seq, int d_model);

        void forward(const MatrixXd& input) override;

        void backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        const MatrixXd& get_output() const override;
        const MatrixXd& get_d_input() const override;

        string get_activation_name() const override;
        LayerType get_type() const override;
};

#endif