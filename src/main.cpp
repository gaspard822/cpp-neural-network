#include <iostream>
#include <fstream>
#include "mlp/neural_network.hpp"
#include "core/loss_function.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "core/relu.hpp"
#include "core/identity.hpp"
#include "core/mean_squared_error_loss.hpp"
#include "core/cross_entropy_loss.hpp"
#include "mlp/mnist.cpp"
#include "transformer/translation.cpp"

using namespace std;


int copy_file(int n) {
    const string in_path  = "../translation/en-fr.csv";
    const string out_path = "../translation/en-fr-short.csv";

    ifstream in(in_path);
    if (!in) { cerr << "Cannot open input\n"; return 1; }

    ofstream out(out_path);
    if (!out) { cerr << "Cannot open output\n"; return 1; }

    string line;

    // Copy header
    if (!getline(in, line)) { cerr << "Empty file\n"; return 1; }
    out << line << "\n";

    // Copy first N rows
    int copied = 0;
    while (copied < n && getline(in, line)) {
        out << line << "\n";
        ++copied;
    }

    cout << "Wrote " << copied << " rows to " << out_path << "\n";
    return 1;
}


int main() {

    // train_test_mnist();
    // infer_mnist();

    // copy_file(100000);
    // doing_stuff();
    train_test_translation();

    return 0;
}