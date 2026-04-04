#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include "mlp/neural_network.hpp"
#include "core/loss_function.hpp"
#include "mlp/fully_connected_layer.hpp"
#include "core/relu.hpp"
#include "core/sigmoid.hpp"
#include "core/identity.hpp"
#include "core/mean_squared_error_loss.hpp"
#include "core/cross_entropy_loss.hpp"
#include "core/adam_optimizer.hpp"
#include "core/vanilla_sgd_optimizer.hpp"

using namespace std;
namespace mx = mlx::core;

struct MNISTData {
    mx::array images;
    mx::array oneHotLabels;
};

/**
 * Reads the csv file containing the training data and returns a struct of type MNISTData that contains the normalized
 * images in a matrix (num_samples x num_pixels) and a matrix with the correct labels (num_samples x 10).
 * The data is provided at https://www.kaggle.com/competitions/digit-recognizer/data.
 */
MNISTData get_mnist_supervised_data(const int first_row, const int last_row) {
    const string filename = "../digit-recognizer/train.csv";
    const int num_rows = last_row - first_row;
    const int num_features = 784;
    const int num_classes = 10;

    vector<float> data(num_rows * num_features);
    vector<float> labels(num_rows * num_classes, 0.0f);

    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Failed to open the file.\n";
        throw runtime_error("Failed to open the file.");
    }

    // Skip header
    getline(file, line);
    // Skip the first first_row lines
    for (int i = 0; i < first_row; i++) {
        getline(file, line);
    }

    int row = 0;
    while (getline(file, line) && row < num_rows) {
        const char* str = line.c_str();
        char* end;

        // Parse label
        int label = strtol(str, &end, 10);
        if (label < 0 || label >= num_classes) {
            throw runtime_error("Invalid label encountered.");
        }
        labels[row * num_classes + label] = 1.0f;
        str = end + 1;

        // Parse pixel values
        for (int col = 0; col < num_features; col++) {
            data[row * num_features + col] = strtof(str, &end) / 255.0f;
            str = end + 1;
        }

        row++;
    }

    file.close();

    mx::array images = mx::array(data.data(), {num_rows, num_features}, mx::float32);
    mx::array oneHotLabels = mx::array(labels.data(), {num_rows, num_classes}, mx::float32);
    return {images, oneHotLabels};
}

/**
 * Reads the csv file containing the testing data and returns the normalized images in a matrix (num_samples x num_pixels).
 */
mx::array get_mnist_testing_data(const int first_row, const int last_row) {
    const string filename = "../digit-recognizer/test.csv";
    const int num_rows = last_row - first_row;
    const int num_features = 784;

    vector<float> data(num_rows * num_features);

    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Failed to open the file.\n";
        throw runtime_error("Failed to open the file.");
    }

    // Skip header
    getline(file, line);
    // Skip the first first_row lines
    for (int i = 0; i < first_row; i++) {
        getline(file, line);
    }

    int row = 0;
    while (getline(file, line) && row < num_rows) {
        const char* str = line.c_str();
        char* end;

        // Parse pixel values
        for (int col = 0; col < num_features; col++) {
            data[row * num_features + col] = strtof(str, &end) / 255.0f;
            str = end + 1;
        }

        row++;
    }

    file.close();
    return mx::array(data.data(), {num_rows, num_features}, mx::float32);
}

/**
 * Takes matrices of the images and the corresponding labels as arguments, as well as references to image and label
 * matrices used for training, validation and testing. Randomly splits the dataset into these according to the specified
 * sizes.
 */
void randomly_split_dataset(const mx::array& all_data, const mx::array& all_labels,
                   mx::array& X_train, mx::array& Y_train,
                   mx::array& X_val,   mx::array& Y_val,
                   mx::array& X_test,  mx::array& Y_test,
                   int train_size, int val_size, int test_size) {

    int num_samples = all_data.shape(0);
    if (train_size + val_size + test_size != num_samples) {
        throw invalid_argument("Sum of split sizes must equal the total number of samples");
    }

    vector<int> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

    mx::array train_idx = mx::array(indices.data(), {train_size}, mx::int32);
    mx::array val_idx = mx::array(indices.data() + train_size, {val_size}, mx::int32);
    mx::array test_idx = mx::array(indices.data() + train_size + val_size, {test_size}, mx::int32);

    X_train = mx::take(all_data, train_idx, 0);
    Y_train = mx::take(all_labels, train_idx, 0);
    X_val = mx::take(all_data, val_idx, 0);
    Y_val = mx::take(all_labels, val_idx, 0);
    X_test = mx::take(all_data, test_idx, 0);
    Y_test = mx::take(all_labels, test_idx, 0);
}

/**
 * Trains and tests a network using the training data provided Kaggle.
 * (https://www.kaggle.com/competitions/digit-recognizer/data)
 */
void train_test_mnist() {
    // Get all data
    MNISTData mnist = get_mnist_supervised_data(0, 42000);
    cout << "Image matrix shape: " << mnist.images.shape(0) << " x " << mnist.images.shape(1) << "\n";
    cout << "Label matrix shape: " << mnist.oneHotLabels.shape(0) << " x " << mnist.oneHotLabels.shape(1) << "\n";

    // Split the dataset into training, validation and testing sets
    auto split_time_start = chrono::high_resolution_clock::now();
    mx::array X_train = mx::zeros({1}, mx::float32);
    mx::array Y_train = mx::zeros({1}, mx::float32);
    mx::array X_val = mx::zeros({1}, mx::float32);
    mx::array Y_val = mx::zeros({1}, mx::float32);
    mx::array X_test = mx::zeros({1}, mx::float32);
    mx::array Y_test = mx::zeros({1}, mx::float32);
    randomly_split_dataset(mnist.images, mnist.oneHotLabels,
                           X_train, Y_train, X_val, Y_val, X_test, Y_test,
                           34000, 4000, 4000);
    auto split_time_end = chrono::high_resolution_clock::now();
    cout << "Time for splitting the data: " << chrono::duration_cast<chrono::milliseconds>(split_time_end - split_time_start).count() << "ms" << endl;

    MultiLayerPerceptronNetwork mlp("CrossEntropy", "Adam");

    // Create layers and add them to the network
    FullyConnectedLayer* layer_1 = new FullyConnectedLayer(new Relu(), 784, 512);
    mlp.add_layer(layer_1);
    FullyConnectedLayer* layer_2 = new FullyConnectedLayer(new Relu(), 512, 256);
    mlp.add_layer(layer_2);
    FullyConnectedLayer* layer_3 = new FullyConnectedLayer(new Relu(), 256, 128);
    mlp.add_layer(layer_3);
    FullyConnectedLayer* layer_4 = new FullyConnectedLayer(new Identity(), 128, 10);
    mlp.add_layer(layer_4);
    mlp.get_optimizer()->update_optimizer();
    
    // Train
    mlp.train(X_train, Y_train, 300, 1024, X_val, Y_val, false);

    mx::array inference = mlp.infer(X_test);
    int num_samples = X_test.shape(0);
    int correct = mx::sum(mx::equal(mx::argmax(inference, 1), mx::argmax(Y_test, 1))).item<int>();
    cout << "Accuracy: " << 100.0 * (float) correct / (float) num_samples << "%" << endl;

    // Save the trained model
    mlp.save_model("../models/testing_stuff.txt");
}

/**
 * Infers the digits on the testing data provided Kaggle using the specified network and creates csv file with the
 * predictions.
 * (https://www.kaggle.com/competitions/digit-recognizer/data)
 */
void infer_mnist() {
    int num_samples = 28000;
    mx::array mnist_data = get_mnist_testing_data(0, num_samples);
    MultiLayerPerceptronNetwork mlp;
    mlp.load_model("../models/testing_stuff.txt");

    mx::array inference = mlp.infer(mnist_data);
    cout << "(" << inference.shape(0) << "x" << inference.shape(1) << ")" << endl;
    mx::array predictions = mx::argmax(inference, 1);
    mx::eval(predictions);
    ofstream file("../models/predictions.csv");
    file << "ImageId,Label" << endl;
    for (int i = 0; i < num_samples; i++) {
        file << i + 1 << "," << mx::slice(predictions, {i}, {i + 1}).item<int>() << "\n";
    }
}
