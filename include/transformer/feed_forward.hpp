#ifndef FEED_FORWARD_HPP
#define FEED_FORWARD_HPP

#include "core/layer.hpp"
#include "core/activation_function.hpp"

class FeedForward: public Layer {
    private:
        MatrixXd X, U, H; // intermediary computations needed for the backpropagation
        mlx::core::array X_mlx, U_mlx, H_mlx;
        MatrixXd W1, W2, d_W1, d_W2;
        mlx::core::array W1_mlx, W2_mlx, d_W1_mlx, d_W2_mlx;
        RowVectorXd b1, b2, d_b1, d_b2;
        mlx::core::array b1_mlx, b2_mlx, d_b1_mlx, d_b2_mlx;
        vector<TrainableParameter> params;
        ActivationFunction* activation;
        int seq, d_model, d_ff;

    public:
        FeedForward(ActivationFunction* activation, int seq, int d_model, int d_ff);

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