#ifndef RELU_HPP
#define RELU_HPP

#include "core/activation_function.hpp"


/**
 * Implements the ReLU (Rectified Linear Unit) activation function.
 */
class Relu : public ActivationFunction {
    public:
        /**
         * Applies the ReLU activation function to the input matrix.
         * @param z The input matrix
         * @return mlx::core::array The result of applying ReLU element-wise
         */
        mlx::core::array apply(const mlx::core::array& z) const override;

        /**
         * Computes the derivative of the ReLU function.
         * @param z The input matrix
         * @return mlx::core::array The element-wise derivative of ReLU
         */
        mlx::core::array derivative(const mlx::core::array& z) const override;

        /**
         * Returns the activation type (ReLU).
         * @return ActivationType Enum value for ReLU
         */
        ActivationType get_type() const override;

        /**
         * Returns the name of the activation function.
         * @return string Activation name
         */
        std::string get_activation_name() const override;
};

#endif
