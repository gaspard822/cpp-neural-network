#include <iostream>
#include "core/cross_entropy_loss.hpp"

using namespace std;
namespace mx = mlx::core;

float CrossEntropy::compute(const mx::array& y_true, const mx::array& y_pred) const {
    // Shift logits for numerical stability
    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array shifted = y_pred - z_max;

    // Compute log-sum-exp
    mx::array log_sum_exp = mx::log(mx::sum(mx::exp(shifted), 1, true));

    // Compute loss: -z_y + z_max + log(sum(exp))
    mx::array true_logits = mx::sum(y_true * y_pred, 1, true);
    return mx::mean(-true_logits + z_max + log_sum_exp).item<float>();
}

mx::array CrossEntropy::derivative(const mx::array& y_true, const mx::array& y_pred) const {
    // Shift logits for numerical stability
    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array exp_shifted = mx::exp(y_pred - z_max);
    mx::array softmax = exp_shifted / mx::sum(exp_shifted, 1, true);
    return (softmax - y_true) / y_true.shape(0);
}

// Does the same computation as the other CrossEntropy::compute(), but doesn't need one-hot enoding
float CrossEntropy::compute(const vector<int>& y_true, const mx::array& y_pred) const {
    int num_tokens = y_true.size();
    int vocab_size = y_pred.shape(1);

    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array shifted = y_pred - z_max;
    mx::array log_sum_exp = mx::log(mx::sum(mx::exp(shifted), 1, true));

    vector<int> flat_indices(num_tokens);
    for (int t = 0; t < num_tokens; t++) flat_indices[t] = t * vocab_size + y_true[t];
    mx::array flat_idx = mx::array(flat_indices.data(), {num_tokens}, mx::int32);
    mx::array true_logits = mx::take(mx::reshape(y_pred, {num_tokens * vocab_size}), flat_idx);

    mx::array loss = -true_logits + mx::reshape(z_max, {num_tokens}) + mx::reshape(log_sum_exp, {num_tokens});
    return mx::mean(loss).item<float>();
}

mx::array CrossEntropy::derivative(const vector<int>& y_true, const mx::array& y_pred) const {
    int num_tokens = y_true.size();
    int vocab_size = y_pred.shape(1);

    mx::array z_max = mx::max(y_pred, 1, true);
    mx::array exp_shifted = mx::exp(y_pred - z_max);
    mx::array softmax = exp_shifted / mx::sum(exp_shifted, 1, true);

    vector<float> one_hot_data(num_tokens * vocab_size, 0.0f);
    for (int t = 0; t < num_tokens; t++) one_hot_data[t * vocab_size + y_true[t]] = 1.0f;
    mx::array one_hot = mx::array(one_hot_data.data(), {num_tokens, vocab_size}, mx::float32);

    return (softmax - one_hot) / num_tokens;
}


float CrossEntropy::compute(const mx::array& y_true_ids, const mx::array& y_pred, int pad_id) const {
    int batch_size = y_pred.shape(0);
    int num_tokens = y_pred.shape(1);
    int vocab_size = y_pred.shape(2);

    // Numerical stability
    mx::array z_max = mx::max(y_pred, -1, true);  // {batch_size, num_tokens, 1}
    mx::array shifted = y_pred - z_max;
    mx::array log_sum_exp = mx::log(mx::sum(mx::exp(shifted), -1, true));  // {batch_size, num_tokens, 1}

    // Extract true-class logits via flatten + take
    mx::array one_hot = mx::astype(mx::equal(mx::arange(0, vocab_size, mx::int32), mx::expand_dims(y_true_ids, -1)), mx::float32);
    mx::array true_logits = mx::sum(y_pred * one_hot, -1);  // {batch_size, num_tokens}

    // Per-token loss
    mx::array per_token_loss = -true_logits + mx::reshape(z_max, {batch_size, num_tokens}) + mx::reshape(log_sum_exp, {batch_size, num_tokens});

    // Mask out padding, average over non-pad tokens
    mx::array mask = mx::astype(mx::not_equal(y_true_ids, mx::array(pad_id)), mx::float32);
    return (mx::sum(per_token_loss * mask) / mx::sum(mask)).item<float>();
}

mx::array CrossEntropy::derivative(const mx::array& y_true_ids, const mx::array& y_pred, int pad_id) const {
    int vocab_size = y_pred.shape(2);

    // Softmax along last axis
    mx::array z_max = mx::max(y_pred, -1, true);
    mx::array exp_shifted = mx::exp(y_pred - z_max);
    mx::array softmax = exp_shifted / mx::sum(exp_shifted, -1, true); // {batch_size, num_tokens, vocab_size}

    // One-hot from integer IDs: compare each vocab index against the target
    mx::array one_hot = mx::astype(
        mx::equal(mx::arange(0, vocab_size, mx::int32), mx::expand_dims(y_true_ids, -1)),
        mx::float32); // {batch_size, num_tokens, vocab_size}

    mx::array grad = softmax - one_hot;

    // Zero out padding positions, divide by non-pad count
    mx::array mask = mx::astype(mx::not_equal(y_true_ids, mx::array(pad_id)), mx::float32); // {batch_size, num_tokens}
    grad = grad * mx::expand_dims(mask, -1);
    return grad / mx::sum(mask);
}

string CrossEntropy::get_loss_name() const {
    return "cross-entropy";
}

LossFunctionType CrossEntropy::get_type() const {
    return LossFunctionType::CROSSENTROPY;
}
