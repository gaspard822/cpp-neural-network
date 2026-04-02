#include <iostream>
#include "core/cross_entropy_loss.hpp"

using namespace std;
namespace mx = mlx::core;

double CrossEntropy::compute(const mx::array& y_true, const mx::array& y_pred) const {
    // Shift logits for numerical stability
    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array shifted = y_pred - z_max;

    // Compute log-sum-exp
    mx::array log_sum_exp = mx::log(mx::sum(mx::exp(shifted), 1, true));

    // Compute loss: -z_y + z_max + log(sum(exp))
    mx::array true_logits = mx::sum(y_true * y_pred, 1, true);
    return mx::mean(-true_logits + z_max + log_sum_exp).item<double>();
}

mx::array CrossEntropy::derivative(const mx::array& y_true, const mx::array& y_pred) const {
    // Shift logits for numerical stability
    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array exp_shifted = mx::exp(y_pred - z_max);
    mx::array softmax = exp_shifted / mx::sum(exp_shifted, 1, true);
    return (softmax - y_true) / y_true.shape(0);
}

// Does the same computation as the other CrossEntropy::compute(), but doesn't need one-hot enoding
double CrossEntropy::compute(const vector<int>& y_true, const mx::array& y_pred) const {
    int T = y_true.size();
    int V = y_pred.shape(1);

    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array shifted = y_pred - z_max;
    mx::array log_sum_exp = mx::log(mx::sum(mx::exp(shifted), 1, true));

    vector<int> flat_indices(T);
    for (int t = 0; t < T; t++) flat_indices[t] = t * V + y_true[t];
    mx::array flat_idx = mx::array(flat_indices.data(), {T}, mx::int32);
    mx::array true_logits = mx::take(mx::reshape(y_pred, {T * V}), flat_idx);

    mx::array loss = -true_logits + mx::reshape(z_max, {T}) + mx::reshape(log_sum_exp, {T});
    return mx::mean(loss).item<double>();
}

mx::array CrossEntropy::derivative(const vector<int>& y_true, const mx::array& y_pred) const {
    int T = y_true.size();
    int V = y_pred.shape(1);

    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array exp_shifted = mx::exp(y_pred - z_max);
    mx::array softmax = exp_shifted / mx::sum(exp_shifted, 1, true);

    vector<float> one_hot_data(T * V, 0.0f);
    for (int t = 0; t < T; t++) one_hot_data[t * V + y_true[t]] = 1.0f;
    mx::array one_hot = mx::array(one_hot_data.data(), {T, V}, mx::float32);

    return (softmax - one_hot) / T;
}


string CrossEntropy::get_loss_name() const {
    return "cross-entropy";
}

LossFunctionType CrossEntropy::get_type() const {
    return LossFunctionType::CROSSENTROPY;
}
