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

    # *** START CODE HERE ***
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE

    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t',  add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c

    clf = LogisticRegression()
    clf.fit(x_train, t_train)

    util.plot(x_train, t_train, theta=clf.theta)
    print("Theta is: ", clf.theta)
    print("The accuracy on training set is: ", np.mean(clf.predict(x_train) == t_train))

    util.plot(x_test, t_test, clf.theta)
    t_pred = clf.predict(x_test)
    print("The accuracy on the test set is: ", np.mean(t_pred == t_test))
    np.savetxt(pred_path, t_test, '%d')

if __name__ == '__main__':
    main(ds3_train_path, ds3_valid_path, ds3_test_path, ds3_pred_path)
