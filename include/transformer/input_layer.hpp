#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "core/layer.hpp"

#define SOS_ID 0
#define EOS_ID 1
#define PAD_ID 2

class InputLayer: public Layer {
    private:
        int seq, d_model, vocab_size;
        string path;
        MatrixXd embeddings;

    public:
        InputLayer(int seq, int d_model, const string& path);

        void forward(const MatrixXd& input) override;

        MatrixXd backward(const MatrixXd& d_output) override;

        MatrixXd infer(const MatrixXd& layer_input) const override;

        unique_ptr<Gradients> get_gradients() override;
        unique_ptr<Gradients> get_params() override;
        string get_activation_name() const override;
        LayerType get_type() const override;
};

#endif