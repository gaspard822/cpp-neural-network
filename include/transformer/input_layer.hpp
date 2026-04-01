#ifndef INPUT_LAYER_HPP
#define INPUT_LAYER_HPP

#include "core/layer.hpp"

class InputLayer: public Layer {
    private:
        int seq, d_model, vocab_size;
        MatrixXd embeddings, d_embeddings;
        mlx::core::array embeddings_mlx, d_embeddings_mlx;
        MatrixXd positional_encodings;
        mlx::core::array positional_encodings_mlx;
        vector<TrainableParameter> params;
        // TODO: MLX-compatible version for TrainableParameter
        
        vector<int> token_ids;
        MatrixXd output;

    public:
        InputLayer(int seq, int d_model, int vocab_size);

        MatrixXd compute_positional_encodings(int seq, int d_model);
        mlx::core::array compute_positional_encodings_mlx(int seq, int d_model);

        void forward(const MatrixXd& input) override;
        void forward(const vector<int>& token_ids);
        void forward_mlx(const vector<int>& token_ids);

        void backward(const MatrixXd& d_output) override;
        void backward_mlx(const mlx::core::array& d_output);

        MatrixXd infer(const MatrixXd& layer_input) const override;
        MatrixXd infer(const vector<int>& token_ids) const;
        mlx::core::array infer_mlx(const vector<int>& token_ids) const;

        const vector<TrainableParameter>& get_parameters() const override;
        const MatrixXd& get_output() const override;
        const mlx::core::array& get_output_mlx() const;
        const MatrixXd& get_d_input() const override;

        string get_layer_name() const override;
        string get_activation_name() const override;
        LayerType get_type() const override;

        void save(ofstream& file) const override;
        void load(ifstream& file) override;
};

#endif