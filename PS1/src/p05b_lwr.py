import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel

ds5_train_path = './PS1/data/ds5_train.csv'
ds5_valid_path = './PS1/data/ds5_valid.csv'
ds5_plot_path = './PS1/output/p05b_plot.png'

def main(tau, train_path, valid_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # Fit a LWR model
    reg = LocallyWeightedLinearRegression(tau)
    reg.fit(x_train, y_train)

    # Get MSE value on the validation set
    y_pred = reg.predict(x_valid)
    mse = np.mean((y_pred - y_valid) ** 2)
    print("The MSE on the validation set is:", mse)

    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_valid, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(ds5_plot_path)

class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m, n = x.shape
        y = np.zeros(m)

        for i in range(m):
            w = np.exp(- np.sum((self.x - x[i]) ** 2, axis=1) / 2 / self.tau ** 2)
            xTw = self.x.T * w
            theta = np.linalg.inv(xTw @ self.x) @ xTw @ self.y
            y[i] = np.dot(theta, x[i])

        return y

if __name__ == '__main__':
    main(0.5, ds5_train_path, ds5_valid_path)
