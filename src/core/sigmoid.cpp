#include <iostream>
#include "core/sigmoid.hpp"

using namespace std;
namespace mx = mlx::core;

mx::array Sigmoid::apply(const mx::array& z) const {
    return mx::array(1.0f) / (mx::array(1.0f) + mx::exp(-z));
}

mx::array Sigmoid::derivative(const mx::array& z) const {
    mx::array s = apply(z);
    return s * (mx::array(1.0f) - s);
}

ActivationType Sigmoid::get_type() const {
    return ActivationType::SIGMOID;
}

string Sigmoid::get_activation_name() const {
    return "Sigmoid";
}
