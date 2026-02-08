#include "transformer/feed_forward.hpp"

FeedForward::FeedForward(ActivationFunction* activation,
                         int seq, int d_model, int d_ff) : activation(activation), seq(seq), d_model(d_model), d_ff(d_ff) {
    
    W1 = MatrixXd::Random(d_model, d_ff);
    W2 = MatrixXd::Random(d_ff, d_model);
    if (activation->get_type() == ActivationType::RELU) {
        // He initialization for the weights if using a ReLU activation function
        // This is not the true He initialization as the weights are chosen from a uniform distribution and not a
        // Gaussian one, but it works well in practice and is efficient
        W1 = W1 * sqrt(2.0 / d_model);
        W2 = W2 * sqrt(2.0 / d_ff);
    } else if (activation->get_type() == ActivationType::SIGMOID) {
        // Glorot initialization for the weights if using a sigmoid activation function
        double limit = sqrt(6.0 / (d_ff + d_model));
        W1 = W1 * limit;
        W2 = W2 * limit;
    }

    b1 = RowVectorXd::Zero(d_ff);
    b2 = RowVectorXd::Zero(d_model);
    
    d_W1 = MatrixXd(d_model, d_ff);
    d_W2 = MatrixXd(d_ff, d_model);
    d_b1 = RowVectorXd(d_ff);
    d_b2 = RowVectorXd(d_model);
}

void FeedForward::forward(const MatrixXd& input) {
    // input : (seq, d_model)
    X = input;
    U = (input * W1).rowwise() + b1;
    H = activation->apply(U);
    output = ((H * W2).rowwise() + b2) + input;
}

void FeedForward::backward(const MatrixXd& d_output) {
    d_W2 = H.transpose() * d_output;
    d_b2 = d_output.colwise().sum();
    MatrixXd d_U = (d_output * W2.transpose()).cwiseProduct(activation->derivative(U));
    d_W1 = X.transpose() * d_U;
    d_b1 = d_U.colwise().sum();
    d_input = d_U * W1.transpose() + d_output;
}

MatrixXd FeedForward::infer(const MatrixXd& input) const {
    return MatrixXd();
}

unique_ptr<Gradients> FeedForward::get_gradients() {
    return nullptr;
}

unique_ptr<Gradients> FeedForward::get_params() {
    return nullptr;
}

const MatrixXd& FeedForward::get_output() const {
    return output;
}

const MatrixXd& FeedForward::get_d_input() const {
    return d_input;
}

string FeedForward::get_activation_name() const {
    return "";
}

LayerType FeedForward::get_type() const {
    return LayerType::FEED_FORWARD;
}
