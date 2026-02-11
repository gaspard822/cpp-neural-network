#ifndef FEED_FORWARD_HPP
#define FEED_FORWARD_HPP

#include "core/layer.hpp"
#include "core/activation_function.hpp"

class FeedForward: public Layer {
    private:
        MatrixXd X, U, H; // intermediary computations needed for the backpropagation
        MatrixXd W1, W2, d_W1, d_W2;
        RowVectorXd b1, b2, d_b1, d_b2;
        vector<TrainableParameter> params;
        ActivationFunction* activation;
        int seq, d_model, d_ff;

    public:
        FeedForward(ActivationFunction* activation, int seq, int d_model, int d_ff);

        void forward(const MatrixXd& input) override;

        void backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        const vector<TrainableParameter>& get_parameters() const override;
        const MatrixXd& get_output() const override;
        const MatrixXd& get_d_input() const override;

        string get_activation_name() const override;
        LayerType get_type() const override;
};

#endif