import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

def compute_entropy(posteriors):
    return -np.sum(posteriors * np.log(posteriors + 1e-8), axis=-1)

def compute_unary(posteriors, weight=10):
    # Negative log likelihood
    return -weight * np.log(posteriors + 1e-10)  # Added a small constant to avoid log(0)

def RBF_kernel(sigma, radius):
    ax = np.arange(-radius, radius + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def dense_crf_inference(cnn_output, iterations=5, step_size=0.03, entropy_weight=0.01, unary_weight=1.0, sigma_alpha=2.0, sigma_beta=5.0, radius=1, use_grad=False):
    print(" =========== running CRF cleanup =========== ")
    print("Iterations: ", str(iterations))
    print("Unary potential weight: ", str(unary_weight))
    print("Entropy potential weight: ", str(entropy_weight))
    print("Sigma_alpha: ", str(sigma_alpha))
    print("Sigma beta: ", str(sigma_beta))
    print("Pairwise potential radius: ", str(radius))
    print("Use_grad: ", str(use_grad))

    depth, height, width, num_labels = cnn_output.shape
    unary = compute_unary(cnn_output, unary_weight)
    orig_entropy = compute_entropy(cnn_output)

    RBF_kernel_3d = RBF_kernel(sigma_alpha, radius)

    new_Q = np.copy(unary)  # Initialize with unary potentials
    for _ in range(iterations):
        print("---- iteration " + str(_) + " ----")
        for l in range(1, num_labels):  # Skip the background channel
            cnn_output_label = cnn_output[..., l]
            convolved_output = gaussian_filter(cnn_output_label, sigma=sigma_alpha, mode='constant', cval=0.0)

            for z in tqdm(range(5, depth - 5), desc="Processing", ascii=True):
                for y in range(5, height - 5):
                    for x in range(5, width - 5):

                        pairwise_diff = convolved_output[z-radius:z+radius+1, y-radius:y+radius+1, x-radius:x+radius+1]

                        new_Q[z, y, x, l] -= step_size * (np.sum(pairwise_diff))
                        new_Q[z, y, x, 0] += step_size * 0.5 * np.sum(pairwise_diff)

        # Convert updated potentials back to beliefs
        Q = np.exp(-new_Q)
        Q /= np.sum(Q, axis=-1, keepdims=True)  # Normalize

    return Q

