#ifndef FEED_FORWARD_HPP
#define FEED_FORWARD_HPP

#include "core/layer.hpp"

class FeedForward: public Layer {
    private:

    public:
        FeedForward(int seq, int d_model);

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