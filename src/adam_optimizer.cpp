#include <iostream>
#include "adam_optimizer.hpp"
#include "fully_connected_layer.hpp"
#include "neural_network.hpp"

// For some reason, need to write "std::move" because i get warnings otherwise
OwnedFCGradients::OwnedFCGradients(MatrixXd dw, VectorXd dbi, RowVectorXd dg, RowVectorXd dbe):
    d_weights(std::move(dw)), d_bias(std::move(dbi)), d_gamma(std::move(dg)), d_beta(std::move(dbe)) {}

AdamOptimizer::AdamOptimizer(NeuralNetwork* new_nn, double stepsize, double b1, double b2) :
    Optimizer(new_nn), stepsize(stepsize), b1(b1), b2(b2) {
        if (b1 < 0.0 || b1 >= 1.0 || b2 < 0.0 || b2 >= 1.0) {
            throw invalid_argument("beta1 and beta2 must be in the interval [0, 1)");
        }
        t = 0;
        epsilon = 1e-8;
    }

AdamOptimizer::AdamOptimizer(NeuralNetwork* new_nn) : AdamOptimizer(new_nn, 0.001, 0.9, 0.999) {}

void AdamOptimizer::update_optimizer(Layer* layer) {
    if (layer->get_type() == LayerType::FULLY_CONNECTED_LAYER) {
        unique_ptr<Gradients> gradients = layer->get_gradients();
        FCGradients* grads = gradients.get()->as_fc_gradients();
        if (!grads) throw runtime_error("The gradients of the layer are not stored as FCGradients");

        // We create OwnedFCGradients that contain first and second moment vectors (initialized to 0) according to the
        // dimensions of the parameters in the corresponding layer, and we add them to m and v
        auto m_copy = make_unique<OwnedFCGradients>(
            MatrixXd::Zero(grads->d_weights.rows(), grads->d_weights.cols()),
            VectorXd::Zero(grads->d_bias.size()),
            RowVectorXd::Zero(grads->d_gamma.size()),
            RowVectorXd::Zero(grads->d_beta.size())
        );
        auto v_copy = make_unique<OwnedFCGradients>(
            MatrixXd::Zero(grads->d_weights.rows(), grads->d_weights.cols()),
            VectorXd::Zero(grads->d_bias.size()),
            RowVectorXd::Zero(grads->d_gamma.size()),
            RowVectorXd::Zero(grads->d_beta.size())
        );
        m.push_back(std::move(m_copy));
        v.push_back(std::move(v_copy));
    } else {
        throw runtime_error("The type of layer was not recognized");
    }
}

void AdamOptimizer::update_parameters(int layer_index) const {
    // Get the gradients and the parameters of the layer
    unique_ptr<Gradients> gradients = nn->get_layers()[layer_index]->get_gradients();
    unique_ptr<Gradients> parameters = nn->get_layers()[layer_index]->get_params();

    // If we're looking at the last layer, increment t
    if (layer_index == nn->get_layers().size() - 1) {
        t += 1;
    }

    if (nn->get_layers()[layer_index]->get_type() == LayerType::FULLY_CONNECTED_LAYER) {
        // Get the gradients and the parameters of the fully connected layer
        FCGradients* grads = gradients.get()->as_fc_gradients();
        FCGradients* params = parameters.get()->as_fc_gradients();
        if (!grads || !params) throw runtime_error("The gradients or parameters of layer " + to_string(layer_index) +
            " are not stored as FCGradients");
        
        // Update biased first moment estimate
        OwnedFCGradients* m_fc = m[layer_index].get()->as_owned_fc_gradients();
        if (!m_fc) throw runtime_error("m[" + to_string(layer_index) + "] is not stored as OwnedFCGradients");
        m_fc->d_weights = b1 * m_fc->d_weights + (1-b1) * grads->d_weights;
        m_fc->d_bias = b1 * m_fc->d_bias + (1-b1) * grads->d_bias;
        m_fc->d_gamma = b1 * m_fc->d_gamma + (1-b1) * grads->d_gamma;
        m_fc->d_beta = b1 * m_fc->d_beta + (1-b1) * grads->d_beta;
        
        // Update biased second raw moment estimate
        OwnedFCGradients* v_fc = v[layer_index].get()->as_owned_fc_gradients();
        if (!v_fc) throw runtime_error("v[" + to_string(layer_index) + "] is not stored as OwnedFCGradients");
        v_fc->d_weights = b2 * v_fc->d_weights + (1-b2) * grads->d_weights.array().square().matrix();
        v_fc->d_bias = b2 * v_fc->d_bias + (1-b2) * grads->d_bias.array().square().matrix();
        v_fc->d_gamma = b2 * v_fc->d_gamma + (1-b2) * grads->d_gamma.array().square().matrix();
        v_fc->d_beta = b2 * v_fc->d_beta + (1-b2) * grads->d_beta.array().square().matrix();

        // Compute bias-corrected first and second moment estimates and update the parameters
        double bc1 = 1.0 - pow(b1, t);
        double bc2 = 1.0 - pow(b2, t);
        params->d_weights = params->d_weights -
            stepsize * ((m_fc->d_weights.array() / bc1) /((v_fc->d_weights.array() / bc2).sqrt() + epsilon)).matrix();
        params->d_bias = params->d_bias -
            stepsize * ((m_fc->d_bias.array() / bc1) / ((v_fc->d_bias.array() / bc2).sqrt() + epsilon)).matrix();
        params->d_gamma = params->d_gamma -
            stepsize * ((m_fc->d_gamma.array() / bc1) / ((v_fc->d_gamma.array() / bc2).sqrt() + epsilon)).matrix();
        params->d_beta = params->d_beta -
            stepsize * ((m_fc->d_beta.array() / bc1) / ((v_fc->d_beta.array() / bc2).sqrt() + epsilon)).matrix();
    } else {
        throw runtime_error("The type of layer " + to_string(layer_index) + " was not recognized");
    }

}

OptimizerType AdamOptimizer::get_type() const {
    return OptimizerType::ADAM;
}