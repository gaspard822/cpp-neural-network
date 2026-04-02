#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "core/activation_function.hpp"

/**
 * Implements the sigmoid activation function.
 */
class Sigmoid : public ActivationFunction {
    public:
        /**
         * Applies the sigmoid activation function to the input matrix.
         * @param z The input matrix
         * @return mlx::core::array The result of applying the sigmoid function element-wise
         */
        mlx::core::array apply(const mlx::core::array& z) const override;

        /**
         * Computes the derivative of the sigmoid function.
         * @param z The input matrix
         * @return mlx::core::array The element-wise derivative of the sigmoid function
         */
        mlx::core::array derivative(const mlx::core::array& z) const override;

        /**
         * Returns the activation type (sigmoid).
         * @return ActivationType Enum value for sigmoid
         */
        ActivationType get_type() const override;

        /**
         * Returns the name of the activation function.
         * @return string Activation name
         */
        std::string get_activation_name() const override;
};

#endif
