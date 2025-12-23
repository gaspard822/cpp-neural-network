#include <iostream>
#include "multi_head_attention_layer.hpp"
#include "neural_network.hpp"
#include "fully_connected_layer.hpp"
#include "relu.hpp"

void doing_stuff() {
    NeuralNetwork nn("CrossEntropy", "VanillaSGD");
    /*
    MultiHeadAttentionLayer* mha_layer_1 = new MultiHeadAttentionLayer(2, 4, 2, AttentionMode.ENCODER_SELF, true);
    nn.add_layer(mha_layer_1);
    FullyConnectedLayer* fc_layer_1 = new FullyConnectedLayer(new Relu(), 2, 2);
    MatrixXd m(2, 4);
    m << 1, 2, 7, 1,
         4, 2, 2, 3;
    MatrixXd target(2, 2);
    */
    // cout << nn.forward(m) << endl;
    // mha_layer_1->forward(m);
    // mha_layer_1->backward(m);
}