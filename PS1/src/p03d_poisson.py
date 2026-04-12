import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel

ds4_train_path = './PS1/data/ds4_train.csv'
ds4_valid_path = './PS1/data/ds4_valid.csv'
ds4_pred_path = './PS1/output/p03d_pred.txt'

def plot(y_label, y_pred, title):
    plt.plot(y_label, 'go', label='label')
    plt.plot(y_pred, 'rx', label='prediction')
    plt.suptitle(title, fontsize=12)
    plt.show()

def main(lr, train_path, valid_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # Fit a Poisson Regression model
    reg = PoissonRegression(step_size=lr, max_iter=1000, eps=1e-5)
    reg.fit(x_train, y_train)
    print("Theta is: ", reg.theta)

    y_pred = reg.predict(x_train)
    ss_res = np.sum((y_train - y_pred) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print("R^2 on training set:", r2)
    plot(y_train, y_pred, 'Training Set')

    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    y_pred = reg.predict(x_valid)
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print("R^2 on validation set:", r2)
    plot(y_valid, y_pred, 'Validation Set')
    print("theta:", reg.theta)
    print("theta norm:", np.linalg.norm(reg.theta))
    np.savetxt(pred_path, y_pred, '%f')


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)

        for _ in range(self.max_iter):
            last_theta = np.copy(self.theta)
            self.theta += self.step_size * x.T.dot(y - np.exp(x.dot(self.theta))) / m

            if np.linalg.norm(self.theta - last_theta, 1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return np.exp(x @ self.theta)
        
if __name__ == '__main__':
    main(1e-7, ds4_train_path, ds4_valid_path, ds4_pred_path)