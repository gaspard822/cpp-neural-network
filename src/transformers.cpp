#include <iostream>
#include "multi_head_attention_layer.hpp"

void doing_stuff() {
    MultiHeadAttentionLayer* mha = new MultiHeadAttentionLayer(2, 4, 2);
    MatrixXd m(2, 4);
    m << 1, 2, 7, 1,
         4, 2, 2, 3;
    mha->forward(m);
}