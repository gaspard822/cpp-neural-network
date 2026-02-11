#include <iostream>
#include "core/adam_optimizer.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"

AdamOptimizer::AdamOptimizer(Network* new_network, double stepsize, double b1, double b2) :
    Optimizer(new_network), stepsize(stepsize), b1(b1), b2(b2) {
    
    if (b1 < 0.0 || b1 >= 1.0 || b2 < 0.0 || b2 >= 1.0) {
        throw invalid_argument("beta1 and beta2 must be in the interval [0, 1)");
    }
    t = 0;
    epsilon = 1e-8;
}

AdamOptimizer::AdamOptimizer(Network* new_network) : AdamOptimizer(new_network, 0.001, 0.9, 0.999) {}

AdamState& AdamOptimizer::get_or_create_state(double* key, Index rows, Index cols) const {
    // operator[] inserts default AdamState if missing
    AdamState& st = states[key];

    // Eigen default constructs as 0x0 matrices, ensure correct shape
    if (st.m.rows() != rows || st.m.cols() != cols) {
        st.m = MatrixXd::Zero(rows, cols);
    }
    if (st.v.rows() != rows || st.v.cols() != cols) {
        st.v = MatrixXd::Zero(rows, cols);
    }
    return st;
}

void AdamOptimizer::update_optimizer() {
    // Optional: pre-create state entries for all parameters.
    vector<Layer*> layers = network->get_layers();
    for (Layer* layer : layers) {
        for (const auto& p : layer->get_parameters()) {
            get_or_create_state(p.value_data, p.rows, p.cols);
        }
    }
}

void AdamOptimizer::update_parameters() const {
    t++;
    const double bc1 = 1.0 - pow(b1, t);
    const double bc2 = 1.0 - pow(b2, t);

    vector<Layer*> layers = network->get_layers();
    for (Layer* layer : layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            // Both matrices and vectors appear as MatrixXd views
            MatrixXd W = p.value();
            MatrixXd G = p.grad();

            AdamState& st = get_or_create_state(p.value_data, p.rows, p.cols);

            st.m = b1 * st.m + (1.0 - b1) * G;
            st.v = b2 * st.v + (1.0 - b2) * G.array().square().matrix();

            const MatrixXd m_hat = st.m / bc1;
            const MatrixXd v_hat = st.v / bc2;

            W -= stepsize * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
        }
    }
}

OptimizerType AdamOptimizer::get_type() const {
    return OptimizerType::ADAM;
}