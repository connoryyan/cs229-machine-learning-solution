import numpy as np
import util

from linear_model import LinearModel

ds1_train_path = './PS1/data/ds1_train.csv'
ds1_eval_path = './PS1/data/ds1_valid.csv'
ds1_pred_path = './PS1/output/p01e_pred1.txt'

def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    clf = GDA()
    clf.fit(x_train, y_train)

    # util.plot(x_train, y_train, theta=clf.theta)
    print("Theta is: ", clf.theta)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == y_train))

    util.plot(x_eval, y_eval, clf.theta)
    y_pred = clf.predict(x_eval)
    print("The accuracy on validation set is: ", np.mean(y_pred == y_eval))
    np.savetxt(pred_path, y_pred, '%d')


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape

        phi = np.sum(y) / m
        mu0 = x.T @ (1 - y) / np.sum(1 - y)
        mu1 = x.T @ y / np.sum(y)
        Sigma = np.zeros((n, n))

        for i in range(m):
            xi = x[i]
            if y[i]:
                x_minus_mu = xi - mu1
            else:
                x_minus_mu = xi - mu0
            Sigma += np.outer(x_minus_mu, x_minus_mu)
        Sigma /= m
        Sigma_inv = np.linalg.inv(Sigma)

        theta = Sigma_inv @ (mu1 - mu0)
        theta0 = -(mu1.T @ Sigma_inv @ mu1 - mu0.T @ Sigma_inv @ mu0)/2 - np.log((1 - phi) / phi)

        self.theta = np.hstack([theta0, theta])

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m, n = x.shape
        return np.hstack([np.ones((m, 1)), x]) @ self.theta >= 0
    
main(ds1_train_path, ds1_eval_path, ds1_pred_path)
