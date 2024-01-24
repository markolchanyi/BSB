import numpy as np
from tqdm import tqdm


def compute_entropy(posteriors):
    return -np.sum(posteriors * np.log(posteriors + 1e-8), axis=-1)

def compute_unary(posteriors, weight=10):
    # Negative log likelihood
    return -weight * np.log(posteriors + 1e-10)  # Added a small constant to avoid log(0)

def compute_pairwise(region_center, radius=3, sigma_alpha=1.0, sigma_beta=2.0, use_grad=False):
    z, y, x = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )

    # Spatial pairwise based on Euclidean distance
    spatial_dist = np.sqrt(z**2 + y**2 + x**2)
    spatial_term = np.exp(-spatial_dist**2 / (2 * sigma_alpha**2))

    # Intensity difference and gradient magnitude terms
    gradient_magnitude = np.linalg.norm(np.gradient(region_center), axis=0)
    if use_grad:
        #gradient_term = np.exp(-gradient_magnitude**2 / (2 * sigma_beta**2))
        gradient_term = 1 + gradient_magnitude/sigma_beta
    # Combined pairwise potential
    if use_grad:
        pairwise = spatial_term * gradient_term
    else:
        pairwise = spatial_term
    return pairwise

def dense_crf_inference(cnn_output, iterations=5, step_size=0.03, entropy_weight=0.01, unary_weight=1.0, sigma_alpha=2.0, sigma_beta=5.0, radius=1, use_grad=False):
    print(" =========== running CRF cleanup =========== ")
    print("ierations: ", str(iterations))
    print("unary potential weight: ", str(unary_weight))
    print("entropy potential weight: ", str(entropy_weight))
    print("sigma_alpha: ", str(sigma_alpha))
    print("sigma beta: ", str(sigma_beta))
    print("pairwise potential radius: ", str(radius))
    print("use_grad: ", str(use_grad))

    depth, height, width, num_labels = cnn_output.shape
    unary = compute_unary(cnn_output, unary_weight)
    orig_entropy = compute_entropy(cnn_output)

    pairwise_matrices = {}  # Cache the pairwise matrices

    new_Q = np.copy(unary)  # Initialize with unary potentials
    for _ in range(iterations):
        print("---- iteration " + str(_) + " ----")
        for z in tqdm(range(5, depth - 5),desc="Processing",ascii=True):
            for y in range(5, height - 5):
                for x in range(5, width - 5):
                    local_region = cnn_output[z-radius:z+radius+1, y-radius:y+radius+1, x-radius:x+radius+1, :]
                    for l in range(1, num_labels):  # Skip the background channel

                        if l not in pairwise_matrices:  # Compute pairwise matrix only once for each label
                            pairwise_matrices[l] = compute_pairwise(local_region[...,l], radius, sigma_alpha, sigma_beta, use_grad)
                        #pairwise = compute_pairwise(local_region[...,l], radius, sigma_alpha, sigma_beta, use_grad)

                        positive_diff = np.maximum(0, cnn_output[z, y, x, l] - local_region[..., l])
                        #pairwise_diff = pairwise * positive_diff
                        pairwise_diff = pairwise_matrices[l] * positive_diff

                        #current_entropy = compute_entropy(np.exp(-new_Q[z, y, x, :]))
                        #entropy_change = current_entropy - orig_entropy[z, y, x]
                        #new_Q[z, y, x, l] -= step_size * (np.sum(pairwise_diff, axis=(0, 1, 2)) + entropy_change))
                        new_Q[z, y, x, l] -= step_size * (np.sum(pairwise_diff, axis=(0, 1, 2)))
                        new_Q[z, y, x, 0] += step_size * 0.5 * np.sum(pairwise_diff, axis=(0, 1, 2))

        # Convert updated potentials back to beliefs
        Q = np.exp(-new_Q)
        Q /= np.sum(Q, axis=-1, keepdims=True)  # Normalize

    #return np.argmax(Q, axis=-1)
    return Q
