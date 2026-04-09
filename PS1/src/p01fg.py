import util
import numpy as np
import matplotlib.pyplot as plt

from p01b_logreg import LogisticRegression
from p01e_gda import GDA

ds1_train_path = './PS1/data/ds1_train.csv'
ds1_eval_path = './PS1/data/ds1_valid.csv'
ds1_pred_path = './PS1/output/p01fg_pred1.png'

ds2_train_path = './PS1/data/ds2_train.csv'
ds2_eval_path = './PS1/data/ds2_valid.csv'
ds2_pred_path = './PS1/output/p01fg_pred2.png'

def main(train_path, eval_path, pred_path):
    """ plot the decision boundaries found by logistic regression and GDA

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    clf2 = GDA()
    clf2.fit(x_train, y_train)

    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    clf1 = LogisticRegression()
    clf1.fit(x_train, y_train)

    plot(x_train, y_train, save_path = pred_path, theta_1=clf1.theta, legend_1='Logistic regression', theta_2=clf2.theta, legend_2='GDA')

def plot(x, y, save_path=None, theta_1=None, legend_1=None, theta_2=None, legend_2=None, title=None, correction=1.0):
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta_1[0] / theta_1[2] * correction + theta_1[1] / theta_1[2] * x1)
    plt.plot(x1, x2, c='red', label=legend_1, linewidth=2)

    if theta_2 is not None:
        x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
        x2 = -(theta_2[0] / theta_1[2] * correction + theta_2[1] / theta_2[2] * x1)
        plt.plot(x1, x2, c='black', label=legend_2, linewidth=2)

    plt.xlabel('x1')
    plt.ylabel('x2')
    if legend_1 is not None or legend_2 is not None:
        plt.legend(loc="upper left")
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

if __name__ == '__main__':
    main(ds1_train_path, ds1_eval_path, ds1_pred_path)
    main(ds2_train_path, ds2_eval_path, ds2_pred_path)