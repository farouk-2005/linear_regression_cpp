#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


class Dataset {
public:
    MatrixXd X;
    VectorXd y;

    Dataset(const MatrixXd& X_data, const VectorXd& y_data) {
        X = X_data;
        y = y_data;
    }

    int size() const { return X.rows(); }
    int n_features() const { return X.cols(); }
};


class LinearRegression {
private:
    VectorXd w;
    double b;


    VectorXd mean;
    VectorXd std;

public:
    LinearRegression(int n_features) {
        w = VectorXd::Zero(n_features);
        b = 0.0;
    }


    void normalize(MatrixXd& X) {
        int m = X.cols();

        mean = X.colwise().mean(); //Computes mean of each column (feature)

        MatrixXd centered = X.rowwise() - mean.transpose(); //Subtracts mean from each row

        std.resize(m);
        for (int j = 0; j < m; j++) {
            std(j) = sqrt(centered.col(j).array().square().mean());
            if (std(j) == 0) std(j) = 1;
        }

        X = centered.array().rowwise() / std.transpose().array(); //array is like fatten in python
    }


    MatrixXd transform(const MatrixXd& X) const {
        MatrixXd centered = X.rowwise() - mean.transpose();
        return centered.array().rowwise() / std.transpose().array();
    }


    VectorXd predict(const MatrixXd& X) const {
        return X * w + VectorXd::Ones(X.rows()) * b;
    }


    double compute_loss(const MatrixXd& X, const VectorXd& y, double lambda) const {
        VectorXd error = predict(X) - y;

        double mse = error.squaredNorm() / X.rows();
        double reg = lambda * w.squaredNorm();

        return mse + reg;
    }

    void train(MatrixXd X, const VectorXd& y,
               int epochs, double alpha, double lambda) {

        int n = X.rows();


        normalize(X);

        for (int epoch = 0; epoch < epochs; epoch++) {

            VectorXd y_pred = predict(X);
            VectorXd error = y_pred - y;


            VectorXd dw = (2.0 / n) * X.transpose() * error
                          + 2.0 * lambda * w;

            double db = (2.0 / n) * error.sum();


            w -= alpha * dw;
            b -= alpha * db;

            if (epoch % 100 == 0) {
                cout << "Epoch " << epoch
                     << " | Loss: " << compute_loss(X, y, lambda)
                     << endl;
            }
        }
    }

    void print_params() const {
        cout << "Weights: " << w.transpose()
             << " | Bias: " << b << endl;
    }
};


int main() {

    MatrixXd X(4, 2);
    X << 1,1,
         2,1,
         3,1,
         4,1;

    VectorXd y(4);
    y << 5,7,9,11;

    Dataset data(X, y);

    LinearRegression model(data.n_features());
    model.train(data.X, data.y, 1000, 0.01, 0.1);
    model.print_params();
    MatrixXd test(1, 2);
    test << 5, 1;
    MatrixXd test_norm = model.transform(test);

    cout << "Prediction: " << model.predict(test_norm) << endl;

    return 0;
}
