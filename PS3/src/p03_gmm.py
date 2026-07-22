import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('PS3', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m, n = x.shape
    group = np.random.randint(0, K, m)

    mu = [0] * K
    sigma = [0] * K
    for j in range(K):
        group_j = group == j
        mu[j] = np.mean(x[group_j], axis=0)
        x_mu = x[group_j] - mu[j]
        sigma[j] = x_mu.T @ x_mu / np.sum(group_j)

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.array([1/K]*K)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.zeros((m, K));
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)

def gaussian(x, mu_j, inv_sigma_j, sqrt_det_sigma_j, pi2k2):
    """Vectorized multivariate normal density for all m examples at once.

    Args:
        x: (m, n)
        mu_j: (n,)
        inv_sigma_j: (n, n)
        sqrt_det_sigma_j: scalar
        pi2k2: scalar, (2*pi)^(n/2)

    Returns:
        (m,) array of densities
    """
    diff = x - mu_j # (m, n)
    esum = np.einsum('ij,jk,ik->i', diff, inv_sigma_j, diff)  # (m,)
    return (1.0 / (pi2k2 * sqrt_det_sigma_j)) * np.exp(-0.5 * esum)

def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    m, n = x.shape
    _, k = w.shape

    pi2k2 = (np.pi * 2) ** (n / 2)

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        pxz = np.zeros((m, k))
        for j in range(k):
            inv_sigma = np.linalg.inv(sigma[j])
            sqrt_det_sigma = np.linalg.det(sigma[j]) ** 0.5
            pxz[:, j] = gaussian(x, mu[j], inv_sigma, sqrt_det_sigma, pi2k2) * phi[j]
                
        
        px = pxz.sum(axis = 1).reshape(-1, 1)
        w = pxz / px

        # (2) M-step: Update the model parameters phi, mu, and sigma
        sum_w = w.sum(axis = 0) # (k, )
        phi = sum_w / m

        sum_wx = w.T @ x

        for j in range(k):
            mu[j] = sum_wx[j] / sum_w[j]
        
        for j in range(k):
            diff = x - mu[j] # (m, n)
            sigma[j] = (w[:, j][:, None] * diff).T @ diff / sum_w[j] # (n, n)
            
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = 0
        pxz_new = np.zeros((m, k))
        for j in range(k):
            inv_sigma = np.linalg.inv(sigma[j])
            sqrt_det_sigma = np.linalg.det(sigma[j]) ** 0.5
            pxz_new[:, j] = gaussian(x, mu[j], inv_sigma, sqrt_det_sigma, pi2k2) * phi[j]
        ll = np.sum(np.log(pxz_new.sum(axis=1)))

        it += 1  
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    print(f"Converge in {it} itrations")
    """
    Converge in 164 itrations
    Converge in 168 itrations
    Converge in 119 itrations
    """
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    m, n = x.shape
    _, k = w.shape
    m_tilde, _ = z.shape
    z_flat = z.flatten().astype(int)

    pi2k2 = (np.pi * 2) ** (n / 2)

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        pxz = np.zeros((m, k))
        for j in range(k):
            inv_sigma = np.linalg.inv(sigma[j])
            sqrt_det_sigma = np.linalg.det(sigma[j]) ** 0.5
            pxz[:, j] = gaussian(x, mu[j], inv_sigma, sqrt_det_sigma, pi2k2) * phi[j]
        
        px = pxz.sum(axis = 1).reshape(-1, 1)
        w = pxz / px

        # (2) M-step: Update the model parameters phi, mu, and sigma
        sum_w = w.sum(axis = 0) # (k, )

        sum_wx = w.T @ x

        for j in range(k):
            mask_j = (z_flat == j)
            sum_j = np.sum(mask_j)
            phi[j] = (sum_w[j] + alpha * sum_j) / (m + alpha * m_tilde)
            mu[j] = (sum_wx[j] + alpha * np.sum(x_tilde[mask_j, :], axis=0)) / (sum_w[j] + alpha * sum_j)
            diff = x - mu[j] # (m, n)
            diff_j = x_tilde[mask_j, :] - mu[j]
            sigma[j] = ((w[:, j][:, None] * diff).T @ diff + alpha * diff_j.T @ diff_j) \
                / (sum_w[j] + alpha * sum_j) # (n, n) 
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        prev_ll = ll
        ll = 0
        pxz_new = np.zeros((m, k))
        ll_tilde = 0
        for j in range(k):
            inv_sigma = np.linalg.inv(sigma[j])
            sqrt_det_sigma = np.linalg.det(sigma[j]) ** 0.5
            pxz_new[:, j] = gaussian(x, mu[j], inv_sigma, sqrt_det_sigma, pi2k2) * phi[j]
            mask_j = (z_flat == j)
            if np.any(mask_j):
                dens = gaussian(x_tilde[mask_j], mu[j], inv_sigma, sqrt_det_sigma, pi2k2)
                ll_tilde += np.sum(np.log(dens))

        ll = np.sum(np.log(pxz_new.sum(axis=1))) + alpha * ll_tilde

        it += 1 

        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    print(f"Converge in {it} itrations")
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('PS3','output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        # main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
