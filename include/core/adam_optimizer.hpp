#ifndef ADAM_OPTIMIZER_HPP
#define ADAM_OPTIMIZER_HPP

#include "core/optimizer.hpp"
#include "mlp/neural_network.hpp"
#include "transformer/transformer.hpp"

struct AdamState {
    mlx::core::array m;
    mlx::core::array v;

    // We need a dummy init because mlx::core::array has no default constructor
    AdamState(): m(mlx::core::zeros({1}, mlx::core::float32)), v(mlx::core::zeros({1}, mlx::core::float32)) {}
};


/**
 * Implementation of the Adam optimization algorithm.
 */
class AdamOptimizer : public Optimizer {
    private:
        // Learning rate
        float stepsize;
        // Exponential decay rates for the moment estimates
        float b1, b2;
        // Small constant to prevent division by zero
        float epsilon;
        // Time step (incremented at each parameter update), mutable to allow const update
        mutable int t;

        mutable std::unordered_map<mlx::core::array*, AdamState> states;
        // Ensures that states[key] exists and has matrices with size (rows x cols)
        AdamState& get_or_create_state(mlx::core::array* key, const mlx::core::Shape& shape) const;

    public:
        /**
         * Constructs an Adam optimizer with custom hyperparameters.
         * @param new_network Pointer to the network to optimize
         * @param stepsize Learning rate
         * @param b1 First moment decay rate
         * @param b2 Second moment decay rate
         */
        AdamOptimizer(Network* new_network, float stepsize, float b1, float b2);

        /**
         * Constructs an Adam optimizer with default hyperparameters
         * stepsize=0.001, b1=0.9, b2=0.999.
         * @param new_network Pointer to the neural network to optimize
         */
        AdamOptimizer(Network* new_network);

        ~AdamOptimizer() = default;

        /**
         * Creates first and second moment vectors corresponding to the layers of the network.
         */
        void update_optimizer() override;

        /**
         * Applies Adam parameter update to the layers of the network.
         * See Algorithm 1 in https://arxiv.org/pdf/1412.6980.
         */
        void update_parameters() const override;

        /**
         * Returns the type of optimizer (Adam).
         * @return OptimizerType Enum value corresponding to the optimizer type
         */
        OptimizerType get_type() const override;

        void save(std::ofstream& file) const override;
        void load(std::ifstream& file) override;
};

#endif
