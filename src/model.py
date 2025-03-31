import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=1e-3):
        """
        Initializes the LinearRegression model.

        Args:
            learning_rate (float, optional): The learning rate for gradient descent. Defaults to 1e-3.
        """
        self.predictions = None  # Stores the model's predictions.
        self.weights = None  # Stores the model's weights (coefficients).
        self.bias = None  # Stores the model's bias (intercept).
        self.learning_rate = learning_rate  # Stores the learning rate.

    def fit(self, X, y, epochs=3):
        """
        Trains the Linear Regression model using gradient descent.

        Args:
            X (np.ndarray): Input features (examples x features).
            y (np.ndarray): Target values (examples x 1).
            epochs (int, optional): Number of training epochs. Defaults to 3.
        """
        num_examples = X.shape[0]  # Number of training examples.
        num_features = X.shape[1]  # Number of features.
        self.weights = np.zeros(shape=(num_features, 1))  # Initialize weights to zeros.
        self.bias = 0  # Initialize bias to zero.

        for epoch in range(epochs):  # Loop through the specified number of epochs.
            self.predict(X)  # Calculate predictions for the current epoch.
            dW, db = self.gradient_descent(X, y, num_examples)  # Calculate gradients.
            self.weights -= self.learning_rate * dW  # Update weights using gradient descent.
            self.bias -= self.learning_rate * db  # Update bias using gradient descent.

    def predict(self, X):
        """
        Makes predictions using the trained model.

        Args:
            X (np.ndarray): Input features for prediction.
        """
        self.predictions = np.matmul(X, self.weights) + self.bias  # Calculate predictions (y = Xw + b).
        return self.predictions

    def means_squared_error(self, labels):
        """
        Calculates the Mean Squared Error (MSE).

        Args:
            labels (np.ndarray): Actual target values.

        Returns:
            float: The Mean Squared Error.
        """
        num_examples = labels.shape[0]  # Number of examples.
        return np.sum(np.subtract(self.predictions, labels) ** 2) / num_examples  # Calculate MSE.

    def gradient_descent(self, X, y, num_examples):
        """
        Calculates the gradients for weights and bias.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            num_examples (int): Number of examples.

        Returns:
            tuple: Gradients for weights (dW) and bias (db).
        """
        dWdl = (2 / num_examples) * np.dot(X.T, self.predictions - y)  # Gradient of weights.
        dbdl = (2 / num_examples) * np.sum(self.predictions - y)  # Gradient of bias.
        return dWdl, dbdl

    def score(self, X, y):
        """
        Calculates the R-squared (R2) score.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.

        Returns:
            float: The R-squared score.
        """
        self.predict(X)  # Make predictions.
        y_mean = np.mean(y)  # Calculate the mean of the target values.
        tss = np.sum(np.square(y - y_mean))  # Total sum of squares.
        rss = np.sum(np.square(y - self.predictions))  # Residual sum of squares.
        return 1 - (rss / tss)  # Calculate and return the R-squared score.
