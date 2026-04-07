#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "core/layer.hpp"


class InputLayer: public Layer {
    private:
        int seq, d_model, vocab_size;
        mlx::core::array token_ids;
        mlx::core::array embeddings, d_embeddings;
        mlx::core::array positional_encodings;
        std::vector<TrainableParameter> params;

    public:
        InputLayer(int seq, int d_model, int vocab_size);

        mlx::core::array compute_positional_encodings(int seq, int d_model);

        void forward(const mlx::core::array& input) override;
        void backward(const mlx::core::array& d_output) override;
        mlx::core::array infer(const mlx::core::array& layer_input) const override;

        const std::vector<TrainableParameter>& get_parameters() const override;
        const mlx::core::array& get_output() const override;
        const mlx::core::array& get_d_input() const override;

        std::string get_layer_name() const override;
        std::string get_activation_name() const override;
        LayerType get_type() const override;

        void save(std::ofstream& file) const override;
        void load(std::ifstream& file) override;
};

#endif
