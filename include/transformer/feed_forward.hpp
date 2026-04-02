#ifndef FEED_FORWARD_HPP
#define FEED_FORWARD_HPP

#include "core/layer.hpp"
#include "core/activation_function.hpp"


class FeedForward: public Layer {
    private:
        mlx::core::array X, U, H;  // intermediary computations needed for the backpropagation
        mlx::core::array W1, W2, d_W1, d_W2;
        mlx::core::array b1, b2, d_b1, d_b2;
        vector<TrainableParameter> params;
        ActivationFunction* activation;
        int seq, d_model, d_ff;

    public:
        FeedForward(ActivationFunction* activation, int seq, int d_model, int d_ff);

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
