import numpy as np
import matplotlib.pyplot as plt
import util

from linear_model import LinearModel

ds1_train_path = './problem-sets/PS1/data/ds1_train.csv'
ds1_eval_path = './problem-sets/PS1/data/ds1_valid.csv'
ds1_pred_path = './problem-sets/PS1/out/ds1_pred.txt'

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # plt.figure(figsize=(12, 8))
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)
    # plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)
    # plt.show()

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # util.plot(x_train, y_train, theta=clf.theta)
    print("Theta is: ", clf.theta)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == y_train))

    util.plot(x_eval, y_eval, clf.theta)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_eval) == y_eval))

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,). 
        """

        def h(theta, x):
            """Hypothesis function

            Args:
                theta: Shape (n,).
                x: Training example inputs. Shape (m, n).

            Returns:
                Shape (m,).
            """
            return 1 / (1 + np.exp(- x @ theta))
        
        def gradient(theta, x, y):
            """Gradient of the loss function J

            Args:
                theta: Shape (n,).
                x: Training example inputs. Shape (m, n).
                y: Training example labels. Shape (m,).

            Returns:
                Shape (n,).
            """
            m, n = x.shape
            return 1 / m * ((h(theta, x) - y) @ x)
        
        def H(theta, x, y):
            """Hessian of the loss function J

            Args:
                theta: Shape (n,).
                x: Training example inputs. Shape (m, n).
                y: Training example labels. Shape (m,).

            Returns:
                Shape (n,n).
            """
            m, n = x.shape
            h_theta_x = np.reshape(h(theta, x), (m, 1))
            return 1 / m * x.T @ ((h_theta_x * (1 - h_theta_x)) * x)
        
        def newtonsMethod(theta, x, y):
            """Use Newton's method to update theta

            Args:
                theta: Shape (n,).
                x: Training example inputs. Shape (m, n).
                y: Training example labels. Shape (m,).

            Returns:
                Shape (n,).
            """
            return theta - np.linalg.inv(H(theta, x, y)) @ gradient(theta, x, y)
        
        m, n = x.shape
        self.theta = np.zeros(n)
        next_theta = newtonsMethod(self.theta, x, y)

        while np.linalg.norm(next_theta - self.theta, 1) >= self.eps:
            self.theta = next_theta
            next_theta = newtonsMethod(self.theta, x, y)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return x @ self.theta >= 0

main(ds1_train_path, ds1_eval_path, ds1_pred_path)