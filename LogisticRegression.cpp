/******************************************************************************
#Logistic Regression using C++

*******************************************************************************/
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));   //sigmoid formula
}

// Logistic regression prediction
double predict(const std::vector<double>& weights, const std::vector<double>& features) {
    double z = weights[0]; // bias term
    for (size_t i = 0; i < features.size(); ++i) {  //Iterate through out feature size
        z += weights[i + 1] * features[i];  
    }
    return sigmoid(z);
}

// gradient descent
std::vector<double> train(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double alpha, int epochs) {
    int n_samples = X.size();
    int n_features = X[0].size();
    std::vector<double> weights(n_features + 1, 0.0); // add bias term

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> gradients(n_features + 1, 0.0);

        for (int i = 0; i < n_samples; ++i) {
            double prediction = predict(weights, X[i]);
            double error = prediction - y[i];

            gradients[0] += error; // bias gradient
            for (int j = 0; j < n_features; ++j) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        // Update weights
        weights[0] -= alpha * gradients[0] / n_samples;
        for (int j = 0; j < n_features; ++j) {
            weights[j + 1] -= alpha * gradients[j + 1] / n_samples;
        }
    }

    return weights;
}

int main() {
    // Seed for random number generation
    std::srand(std::time(0));

    // Sample dataset (4 samples, 2 features)
    std::vector<std::vector<double>> X = {
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 0.0}
    };

    // Labels
    std::vector<int> y = {0, 1, 1, 0};

    // Training parameters
    double alpha = 0.1; // learning rate
    int epochs = 1000;

    // Training the model
    std::vector<double> weights = train(X, y, alpha, epochs);

    // Output the model weights
    std::cout << "Weights:" << std::endl;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::cout << "w" << i << ": " << weights[i] << std::endl;
    }

    // Let's test the model
    std::vector<std::vector<double>> X_test = {
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 0.0}
    };
    std::vector<int> y_test = {0, 1, 1, 0};

    std::cout << "Predictions:" << std::endl;  //Get the prediction values
    for (size_t i = 0; i < X_test.size(); ++i) {  //Iterate through out test dataset size
        double prediction = predict(weights, X_test[i]);
        std::cout << "Sample " << i + 1 << ": " << (prediction >= 0.5 ? 1 : 0) << " (Prob: " << prediction << ")" << std::endl;
    }

    return 0;
}