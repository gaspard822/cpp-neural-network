#ifndef IDENTITY_HPP
#define IDENTITY_HPP

#include "core/activation_function.hpp"


/**
 * Implements the identity activation function.
 * While the identity if not an activation function, it is convenient to represent it as such for implementation.
 */
class Identity : public ActivationFunction {
    public:
        /**
         * Applies the identity activation function to the input matrix.
         * @param z The input matrix
         * @return mlx::core::array A copy of the input matrix
         */
        mlx::core::array apply(const mlx::core::array& z) const override;

        /**
         * Computes the derivative of the identity function.
         * @param z The input matrix
         * @return mlx::core::array A matrix of ones
         */
        mlx::core::array derivative(const mlx::core::array& z) const override;

        /**
         * Returns the activation type (identity).
         * @return ActivationType Enum value for identity
         */
        ActivationType get_type() const override;

        /**
         * Returns the name of the activation function.
         * @return string Activation name
         */
        string get_activation_name() const override;
};

#endif
