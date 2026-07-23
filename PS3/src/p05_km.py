from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
import numpy as np

def k_means(k, x):
    """
    K-Means

    Args:
        k: number of clusters
        x: m*m pixels image with r, g, and b channels, of shape (m, n, 3)
    
    Returns:
        Compressed image's np array of shape (m, n, 3)
    """

    min_iter = 30
    max_iter = 1000
    eps = 1e-3

    m, n, _ = x.shape
    x = x.astype(np.float64)

    flat = x.reshape(-1, 3)
    idx = np.random.choice(flat.shape[0], k, replace=False)
    mu = flat[idx]
    z = np.zeros((m, n), dtype=int)

    it = 0
    prev_mu = None

    while (it < min_iter) or (prev_mu is None or np.linalg.norm(mu - prev_mu) > eps and it < max_iter):
        diff = x[:, :, np.newaxis, :] - mu[np.newaxis, np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=3) # shape (m, n, k)
        z = np.argmin(dist, axis=2) # shape (m, n)

        prev_mu = mu.copy()
        for i in range(k):
            mask = (z == i)
            if np.any(mask):
                mu[i] = np.mean(x[mask], axis=0)

        it += 1
    print(f"Converge in {it} iterations.")

    compressed = mu[z]
    return np.clip(compressed, 0, 255).astype(np.uint8)

def main():
    A = imread('PS3/data/peppers-small.tiff')

    k = 16
    compressed = k_means(k, A)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(A)
    axes[0].set_title('Original')
    axes[0].axis('off')
 
    axes[1].imshow(compressed)
    axes[1].set_title(f'Compressed (k={k})')
    axes[1].axis('off')
 
    plt.tight_layout()
    plt.savefig('PS3/output/peppers_compressed_comparison.png', dpi=150)
    plt.show()
 
    imsave('PS3/output/peppers_compressed.png', compressed)
 
 
if __name__ == '__main__':
    main()

    