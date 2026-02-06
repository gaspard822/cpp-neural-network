#ifndef LAYER_NORM_HPP
#define LAYER_NORM_HPP

#include "core/layer.hpp"

class LayerNorm: public Layer {
    private:

    public:
        LayerNorm(int seq, int d_model);

        void forward(const MatrixXd& input) override;

        MatrixXd backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        string get_activation_name() const override;
        LayerType get_type() const override;
};

#endif