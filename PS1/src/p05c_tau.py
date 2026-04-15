import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression

ds5_train_path = './PS1/data/ds5_train.csv'
ds5_valid_path = './PS1/data/ds5_valid.csv'
ds5_test_path = './PS1/data/ds5_test.csv'
ds5_pred_path = './PS1/output/p05c_pred.txt'
ds5_plot_path = './PS1/output/p05c_plot'

def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    min_mse = 0.3305312682137524 # MSE with tau = 0.5
    best_tau = 0.5
    reg = LocallyWeightedLinearRegression(0.5)
    reg.fit(x_train, y_train)

    # Search tau_values for the best tau (lowest MSE on the validation set)
    for tau in tau_values:
        reg.tau = tau
        y_pred = reg.predict(x_valid)

        plt.figure()
        plt.plot(x_train, y_train, 'bx')
        plt.plot(x_valid, y_pred, 'ro')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'{ds5_plot_path}_tau={tau}.png')

        mse = np.mean((y_pred - y_valid) ** 2)
        if mse < min_mse:
            best_tau = mse

    # Fit a LWR model with the best tau value
    reg.tau = best_tau

    # Run on the test set to get the MSE value
    y_pred = reg.predict(x_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print("The MSE on the test set is:", mse)

    plt.figure()
    plt.plot(x_train, y_train, 'bx')
    plt.plot(x_valid, y_pred, 'ro')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(ds5_plot_path + '_test.png')

    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred, '%f')

if __name__ == '__main__':
    main([3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1], ds5_train_path, ds5_valid_path, ds5_test_path, ds5_pred_path)