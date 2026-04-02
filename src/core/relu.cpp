#include <iostream>
#include "core/relu.hpp"

using namespace std;
namespace mx = mlx::core;

mx::array Relu::apply(const mx::array& z) const {
    return mx::maximum(z, mx::array(0.0f));
}

mx::array Relu::derivative(const mx::array& z) const {
    return mx::where(z > mx::array(0.0f), mx::array(1.0f), mx::array(0.0f));
}

ActivationType Relu::get_type() const {
    return ActivationType::RELU;
}

string Relu::get_activation_name() const {
    return "Relu";
}
