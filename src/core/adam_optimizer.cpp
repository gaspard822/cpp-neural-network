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
    epsilon = 1e-6;
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
            auto weights = p.value();
            auto gradients = p.grad();

            AdamState& st = get_or_create_state(p.value_data, p.rows, p.cols);

            st.m = b1 * st.m + (1.0 - b1) * gradients;
            st.v = b2 * st.v + (1.0 - b2) * gradients.array().square().matrix();

            const MatrixXd m_hat = st.m / bc1;
            const MatrixXd v_hat = st.v / bc2;

            weights -= stepsize * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
        }
    }
}

OptimizerType AdamOptimizer::get_type() const {
    return OptimizerType::ADAM;
}

void AdamOptimizer::save(ofstream& file) const {
    file << t << "\n";
    for (Layer* layer : network->get_layers()) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            auto it = states.find(p.value_data);
            if (it == states.end()) {
                // No state yet (shouldn't happen after training) - write zeros
                for (Index i = 0; i < p.rows * p.cols; i++) file << "0 ";
                file << "\n";
                for (Index i = 0; i < p.rows * p.cols; i++) file << "0 ";
                file << "\n";
            } else {
                const MatrixXd& m = it->second.m;
                const MatrixXd& v = it->second.v;
                for (Index r = 0; r < m.rows(); r++)
                    for (Index c = 0; c < m.cols(); c++)
                        file << m(r, c) << " ";
                file << "\n";
                for (Index r = 0; r < v.rows(); r++)
                    for (Index c = 0; c < v.cols(); c++)
                        file << v(r, c) << " ";
                file << "\n";
            }
        }
    }
}

void AdamOptimizer::load(ifstream& file) {
    file >> t;
    for (Layer* layer : network->get_layers()) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            AdamState& st = get_or_create_state(p.value_data, p.rows, p.cols);
            for (Index r = 0; r < p.rows; r++)
                for (Index c = 0; c < p.cols; c++)
                    file >> st.m(r, c);
            for (Index r = 0; r < p.rows; r++)
                for (Index c = 0; c < p.cols; c++)
                    file >> st.v(r, c);
        }
    }
}
