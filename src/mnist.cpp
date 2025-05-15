#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include "neural_network.hpp"
#include "loss_function.hpp"
#include "fully_connected_layer.hpp"
#include "relu.hpp"
#include "sigmoid.hpp"
#include "identity.hpp"
#include "mean_squared_error_loss.hpp"
#include "cross_entropy_loss.hpp"
#include "adam_optimizer.hpp"
#include "vanilla_sgd_optimizer.hpp"

using namespace std;
using namespace Eigen;

struct MNISTData {
    MatrixXd images;
    MatrixXd oneHotLabels;
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

    MatrixXd data(num_rows, num_features);
    MatrixXd labels = MatrixXd::Zero(num_rows, num_classes);

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
        labels(row, label) = 1.0;
        str = end + 1;

        // Parse pixel values
        for (int col = 0; col < num_features; col++) {
            data(row, col) = strtof(str, &end);
            str = end + 1;
        }

        row++;
    }

    file.close();
    data /= 255.0f;

    return {data, labels};
}

/**
 * Reads the csv file containing the testing data and returns the normalized images in a matrix (num_samples x num_pixels).
 */
MatrixXd get_mnist_testing_data(const int first_row, const int last_row) {
    const string filename = "../digit-recognizer/test.csv";
    const int num_rows = last_row - first_row;
    const int num_features = 784;
    const int num_classes = 10;

    MatrixXd data(num_rows, num_features);
    // MatrixXd labels = MatrixXd::Zero(num_rows, num_classes);

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
            data(row, col) = strtof(str, &end);
            str = end + 1;
        }

        row++;
    }

    file.close();
    data /= 255.0f;

    return data;
}

/**
 * Takes matrices of the images and the corresponding labels as arguments, as well as references to image and label
 * matrices used for training, validation and testing. Randomly splits the dataset into these according to the specified
 * sizes.
 */
void randomly_split_dataset(const MatrixXd& all_data, const MatrixXd& all_labels,
                   MatrixXd& X_train, MatrixXd& Y_train,
                   MatrixXd& X_val,   MatrixXd& Y_val,
                   MatrixXd& X_test,  MatrixXd& Y_test,
                   int train_size,
                   int val_size,
                   int test_size) {

    int num_samples = all_data.rows();
    if (train_size + val_size + test_size != num_samples) {
        throw invalid_argument("Sum of split sizes must equal the total number of samples");
    }

    vector<int> indices(num_samples);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

    X_train = MatrixXd(train_size, all_data.cols());
    Y_train = MatrixXd(train_size, all_labels.cols());
    X_val = MatrixXd(val_size, all_data.cols());
    Y_val = MatrixXd(val_size, all_labels.cols());
    X_test = MatrixXd(test_size, all_data.cols());
    Y_test = MatrixXd(test_size, all_labels.cols());

    // Not the most efficient way but works well, doesn't take too much time and didn't find a cleaner way to do this with Eigen
    for (int i = 0; i < train_size; ++i) {
        X_train.row(i) = all_data.row(indices[i]);
        Y_train.row(i) = all_labels.row(indices[i]);
    }

    for (int i = 0; i < val_size; ++i) {
        X_val.row(i) = all_data.row(indices[train_size + i]);
        Y_val.row(i) = all_labels.row(indices[train_size + i]);
    }

    for (int i = 0; i < test_size; ++i) {
        X_test.row(i) = all_data.row(indices[train_size + val_size + i]);
        Y_test.row(i) = all_labels.row(indices[train_size + val_size + i]);
    }
}

/**
 * Trains and tests a network using the training data provided Kaggle.
 * (https://www.kaggle.com/competitions/digit-recognizer/data)
 */
void train_test_mnist() {
    // Get all data
    MNISTData mnist = get_mnist_supervised_data(0, 42000);
    cout << "Image matrix shape: " << mnist.images.rows() << " x " << mnist.images.cols() << "\n";
    cout << "Label matrix shape: " << mnist.oneHotLabels.rows() << " x " << mnist.oneHotLabels.cols() << "\n";
    
    auto split_time_start = chrono::high_resolution_clock::now();
    // Split the dataset into training, validation and testing sets
    MatrixXd X_train, Y_train, X_val, Y_val, X_test, Y_test;
    randomly_split_dataset(mnist.images, mnist.oneHotLabels,
                           X_train, Y_train, X_val, Y_val, X_test, Y_test,
                           34000, 4000, 4000);
    auto split_time_end = chrono::high_resolution_clock::now();
    cout << "Time for splitting the data: " << chrono::duration_cast<chrono::milliseconds>(split_time_end - split_time_start).count() << "ms" << endl;
    
    // Create a neural network using cross-entropy as a loss function and adam as an optimizer
    NeuralNetwork nn("CrossEntropy", "Adam");

    // Create layers and add them to the network
    FullyConnectedLayer* layer_1 = new FullyConnectedLayer(new Relu(), 784, 512);
    nn.add_layer(layer_1);
    FullyConnectedLayer* layer_2 = new FullyConnectedLayer(new Relu(), 512, 256);
    nn.add_layer(layer_2);
    FullyConnectedLayer* layer_3 = new FullyConnectedLayer(new Relu(), 256, 128);
    nn.add_layer(layer_3);
    FullyConnectedLayer* layer_4 = new FullyConnectedLayer(new Identity(), 128, 10);
    nn.add_layer(layer_4);
    
    // Train
    nn.train(X_train, Y_train, 300, 1024, X_val, Y_val, false, false);

    // Saving the model and loading it again to test the save_model() and load_model() functions
    // Save the trained model
    nn.save_model("../models/testing_stuff.txt");
    // Create a new network and load the architecture and parameters of the previously trained network
    NeuralNetwork nn_test;
    nn_test.load_model("../models/testing_stuff.txt");
    // Infer the testing set
    MatrixXd inference = nn_test.infer(X_test);
    // For each sample, take the index of the logit with the highest value as the prediction
    int num_samples = X_test.rows();
    VectorXd prediction(num_samples);
    VectorXd truth(num_samples);
    int correct_predictions = 0;
    // Not the most efficient way but works well, doesn't take too much time and didn't find a cleaner way to do this with Eigen
    for (int i = 0; i < num_samples; i++) {
        Index pred, label;
        inference.row(i).maxCoeff(&pred);
        Y_test.row(i).maxCoeff(&label);
        if (pred == label) correct_predictions += 1;
    }
    cout << "Accuracy: " << 100.0 * (double) correct_predictions / (double) num_samples << "%" << endl;
}

/**
 * Infers the digits on the testing data provided Kaggle using the specified network and creates csv file with the
 * predictions.
 * (https://www.kaggle.com/competitions/digit-recognizer/data)
 */
void infer_mnist() {
    int num_samples = 28000;
    MatrixXd mnist_data = get_mnist_testing_data(0, num_samples);
    NeuralNetwork nn;
    nn.load_model("../models/testing_stuff.txt");
    
    MatrixXd inference = nn.infer(mnist_data);
    cout << "(" << inference.rows() << "x" << inference.cols() << ")" << endl;
    VectorXd prediction(num_samples);
    ofstream file("../models/predictions.csv");
    file << "ImageId,Label" << endl;
    for (int i = 0; i < num_samples; i++) {
        Index pred;
        inference.row(i).maxCoeff(&pred);
        file << i+1 << "," << pred << "\n";
    }
}