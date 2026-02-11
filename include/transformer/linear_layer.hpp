#ifndef LINEAR_LAYER_HPP
#define LINEAR_LAYER_HPP

#include "core/layer.hpp"

class LinearLayer: public Layer {
    private:
        MatrixXd X;
        MatrixXd W, d_W;
        RowVectorXd b, d_b;
        vector<TrainableParameter> params;
        int d_model, vocab_size;

    public:
        LinearLayer(int d_model, int vocab_size);

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