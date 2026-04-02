#include "core/identity.hpp"

using namespace std;
namespace mx = mlx::core;

mlx::core::array Identity::apply(const mlx::core::array& z) const {
    return z;
}

mlx::core::array Identity::derivative(const mlx::core::array& z) const {
    return mx::ones_like(z);
}

ActivationType Identity::get_type() const {
    return ActivationType::IDENTITY;
}

string Identity::get_activation_name() const {
    return "Identity";
}