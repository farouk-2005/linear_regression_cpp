# Linear Regression in C++ with Eigen

A simple implementation of regularized linear regression from scratch using the [Eigen](https://eigen.tuxfamily.org/) library.

## Features

- Feature normalization (zero mean, unit variance)
- Gradient descent optimization
- L2 regularization (Ridge)
- MSE loss tracking during training

## Requirements

- C++14 or later
- [Eigen 3.x](https://eigen.tuxfamily.org/dox/GettingStarted.html)

## Build

```bash
g++ -std=c++14 -I /path/to/eigen main.cpp -o linear_regression
./linear_regression
```

> Replace `/path/to/eigen` with the actual path to your Eigen headers (e.g. `/usr/include/eigen3`).

## Usage

The `LinearRegression` class exposes three main methods:

| Method | Description |
|---|---|
| `train(X, y, epochs, alpha, lambda)` | Fits the model using gradient descent |
| `predict(X)` | Returns predictions for normalized input |
| `transform(X)` | Normalizes new data using training statistics |

**Example:**

```cpp
LinearRegression model(n_features);
model.train(X, y, /*epochs=*/1000, /*alpha=*/0.01, /*lambda=*/0.1);

MatrixXd test(1, 2);
test << 5, 1;
MatrixXd test_norm = model.transform(test);
cout << model.predict(test_norm) << endl;
```

## Project Structure

```
.
└── main.cpp   # Dataset class, LinearRegression class, and demo main()
```
