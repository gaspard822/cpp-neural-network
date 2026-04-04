#include <iostream>
#include "core/adam_optimizer.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"
#include "core/mlx_utils.hpp"

using namespace std;
namespace mx = mlx::core;

AdamOptimizer::AdamOptimizer(Network* new_network, float stepsize, float b1, float b2) :
    Optimizer(new_network), stepsize(stepsize), b1(b1), b2(b2) {
    
    if (b1 < 0.0 || b1 >= 1.0 || b2 < 0.0 || b2 >= 1.0) {
        throw invalid_argument("beta1 and beta2 must be in the interval [0, 1)");
    }
    t = 0;
    epsilon = 1e-6;
}

AdamOptimizer::AdamOptimizer(Network* new_network) : AdamOptimizer(new_network, 0.001, 0.9, 0.999) {}

AdamState& AdamOptimizer::get_or_create_state(mx::array* key, const mx::Shape& shape) const {
    auto it = states.find(key);
    if (it == states.end()) {
        states[key] = AdamState();
        states[key].m = mx::zeros(shape, mx::float32);
        states[key].v = mx::zeros(shape, mx::float32);
    }
    return states[key];
}

void AdamOptimizer::update_optimizer() {
    vector<Layer*> layers = network->get_layers();
    for (Layer* layer : layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            get_or_create_state(p.value, p.value->shape());
        }
    }
}

void AdamOptimizer::update_parameters() const {
    t++;
    const float bc1 = 1.0f - pow(b1, t);
    const float bc2 = 1.0f - pow(b2, t);

    vector<Layer*> layers = network->get_layers();
    for (Layer* layer : layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            mx::array& weights = *p.value;
            mx::array& gradients = *p.grad;

            AdamState& st = get_or_create_state(p.value, weights.shape());

            st.m = b1 * st.m + (1.0f - b1) * gradients;
            st.v = b2 * st.v + (1.0f - b2) * mx::square(gradients);

            const mx::array m_hat = st.m / bc1;
            const mx::array v_hat = st.v / bc2;

            weights = weights - stepsize * (m_hat / (mx::sqrt(v_hat) + epsilon));
        }
    }

    // Eval the new weights and adam states so that the computation graph doesn't become huge
    vector<mx::array> to_eval;
    for (Layer* layer : layers) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            to_eval.push_back(*p.value);
            AdamState& st = states[p.value];
            to_eval.push_back(st.m);
            to_eval.push_back(st.v);
        }
    }
    mx::eval(to_eval);
}

OptimizerType AdamOptimizer::get_type() const {
    return OptimizerType::ADAM;
}

void AdamOptimizer::save(ofstream& file) const {
    file << t << "\n";
    for (Layer* layer : network->get_layers()) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            auto it = states.find(p.value);
            if (it != states.end()) {
                save_array(file, it->second.m);
                save_array(file, it->second.v);
            } else {
                throw runtime_error("Couldn't find the adam state of the parameters of some " + layer->get_layer_name());
            }
        }
    }
}

void AdamOptimizer::load(ifstream& file) {
    file >> t;
    for (Layer* layer : network->get_layers()) {
        for (const TrainableParameter& p : layer->get_parameters()) {
            AdamState& st = get_or_create_state(p.value, p.value->shape());
            st.m = load_array(file);
            st.v = load_array(file);
        }
    }
}
