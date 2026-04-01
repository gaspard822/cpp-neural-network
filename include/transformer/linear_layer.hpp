#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "core/layer.hpp"

class LinearLayer: public Layer {
    private:
        MatrixXd X;
        mlx::core::array X_mlx;
        MatrixXd W, d_W;
        mlx::core::array W_mlx, d_W_mlx;
        RowVectorXd b, d_b;
        mlx::core::array b_mlx, d_b_mlx;
        vector<TrainableParameter> params;
        int d_model, vocab_size;

    public:
        LinearLayer(int d_model, int vocab_size);

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