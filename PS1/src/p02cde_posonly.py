import numpy as np
import util

from p01b_logreg import LogisticRegression

ds3_train_path = './PS1/data/ds3_train.csv'
ds3_valid_path = './PS1/data/ds3_valid.csv'
ds3_test_path = './PS1/data/ds3_test.csv'
ds3_pred_path = './PS1/output/p02X_pred.txt'

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """

    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    pred_plot_c = pred_path.replace(WILDCARD, 'c').replace('pred.txt', 'plot.png')
    pred_plot_d = pred_path.replace(WILDCARD, 'd').replace('pred.txt', 'plot.png')
    pred_plot_e = pred_path.replace(WILDCARD, 'e').replace('pred.txt', 'plot.png')

    clf = LogisticRegression()

    # ---------- Part (c): Train and test on true labels ----------

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    clf.fit(x_train, t_train)

    # util.plot(x_train, t_train, theta=clf.theta)
    # print("Theta is: ", clf.theta)
    # print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == t_train))

    util.plot(x_test, t_test, clf.theta, pred_plot_c)
    t_pred = clf.predict(x_test)
    # print("The accuracy on the test set is: ", np.mean(t_pred == t_test))
    np.savetxt(pred_path_c, t_pred, '%d')

    # ---------- Part (d): Train on y-labels and test on true labels ---------

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    clf.fit(x_train, y_train)

    util.plot(x_train, y_train, theta=clf.theta)
    print("Theta is: ", clf.theta)

    util.plot(x_test, t_test, clf.theta, pred_plot_d)
    y_pred = clf.predict(x_test)
    np.savetxt(pred_path_d, y_pred, '%d')

    # ---------- Part (e): Apply correction factor using validation set and test on true labels ---------
    
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y',  add_intercept=True)
    x_valid_labelled = x_valid[y_valid == 1]

    alpha = np.mean(1 / (1 + np.exp(-x_valid_labelled @ clf.theta)))
    theta = np.hstack([clf.theta[0] - np.log(alpha / (2 - alpha)), clf.theta[1:]])

    t_pred = (x_valid @ theta >= 0)
    print("The accuracy on the test set is: ", np.mean(t_pred == t_test))
    util.plot(x_test, t_test, theta, pred_plot_e)
    np.savetxt(pred_path_e, t_pred, '%d')

if __name__ == '__main__':
    main(ds3_train_path, ds3_valid_path, ds3_test_path, ds3_pred_path)
