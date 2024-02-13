import numpy as np
from tqdm import tqdm
import tensorflow as tf

def compute_entropy(posteriors):
    return -np.sum(posteriors * np.log(posteriors + 1e-8), axis=-1)

def compute_unary(posteriors, weight=10):
    return -weight * np.log(posteriors + 1e-10)  # Added a small constant to avoid log(0)

def compute_pairwise(region_center, radius=3, sigma_alpha=1.0, sigma_beta=2.0, use_grad=False):
    z, y, x = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )

    # Euclidean distance
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


def RBF_kernel(size=3, sigma=1.0):
    # RBF kernel discretized via a gaussian..
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    y = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    z = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    xx, yy, zz = tf.meshgrid(x, y, z)
    kernel = tf.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel[..., tf.newaxis, tf.newaxis]


def convolve_4d_with_gaussian(vol, size=3, sigma=1.0):
    """Convolves a 4D numpy array with a Gaussian kernel across all spatial dimensions."""
    vol = tf.convert_to_tensor(vol, dtype=tf.float32)
    output_channels = []

    kernel = RBF_kernel(size, sigma)

    for i in range(vol.shape[-1]):
        channel = vol[..., i:i+1]
        channel = tf.expand_dims(channel, axis=0)

        # 3d conv
        convolved_channel = tf.nn.conv3d(channel, kernel, strides=[1, 1, 1, 1, 1], padding="SAME")

        convolved_channel = tf.squeeze(convolved_channel, axis=0)
        convolved_channel = tf.squeeze(convolved_channel, axis=-1)

        output_channels.append(convolved_channel)

    # Stack individual channels
    output_vol = tf.stack(output_channels, axis=-1)
    return output_vol.numpy()



def dense_crf_inference(cnn_output, iterations=5, step_size=0.015, entropy_weight=0.01, unary_weight=1.0, sigma_alpha=2.0, sigma_beta=5.0, sigma_K=1.0, radius=1, use_grad=False, rbf_convolve=True):
    print(" =========== running CRF cleanup =========== ")
    print("ierations: ", str(iterations))
    print("unary potential weight: ", str(unary_weight))
    print("entropy potential weight: ", str(entropy_weight))
    print("sigma_alpha: ", str(sigma_alpha))
    print("sigma beta: ", str(sigma_beta))
    print("RBF sigma: ", str(sigma_K))
    print("pairwise potential region size: ", str(radius*2 + 1))
    print("use_grad: ", str(use_grad))

    depth, height, width, num_labels = cnn_output.shape
    unary = compute_unary(cnn_output, unary_weight)
    orig_entropy = compute_entropy(cnn_output)

    pairwise_matrices = {}  # pairwise potential cache
    new_Q = np.copy(unary)  # Initialize with unary potentials

    for _ in range(iterations):
        print("---- iteration " + str(_) + " ----")

        # approximate distance term of pairwise potential with RBF each iteration
        # kernel size should be the same size as the local region for pairwise pot.
        if rbf_convolve:
            new_Q_rbf_convolved = convolve_4d_with_gaussian(new_Q, size=(radius*2+1), sigma=sigma_K)

        for z in tqdm(range(5, depth - 5),desc="Processing",ascii=True):
            for y in range(5, height - 5):
                for x in range(5, width - 5):
                    if rbf_convolve:
                        local_region = new_Q_rbf_convolved[z-radius:z+radius+1, y-radius:y+radius+1, x-radius:x+radius+1, :]
                        for l in range(1, num_labels):  # Skip the background channel
                            pairwise_diff = np.maximum(0, new_Q_rbf_convolved[z, y, x, l] - local_region[..., l])
                            new_Q[z, y, x, l] -= step_size * (np.sum(pairwise_diff, axis=(0, 1, 2)))

                        # apply opposite pairwise energy penalty to background label just to avoid background inpainting/dominance
                        new_Q[z, y, x, 0] += step_size * 0.5 * np.sum(pairwise_diff, axis=(0, 1, 2))

                    else:
                        local_region = new_Q[z-radius:z+radius+1, y-radius:y+radius+1, x-radius:x+radius+1, :]
                        for l in range(1, num_labels):  # Skip the background channel
                            if l not in pairwise_matrices:  # Compute pairwise matrix only once for each label
                                pairwise_matrices[l] = compute_pairwise(local_region[...,l], radius, sigma_alpha, sigma_beta, use_grad)

                            positive_diff = np.maximum(0, new_Q[z, y, x, l] - local_region[..., l])
                            pairwise_diff = pairwise_matrices[l] * positive_diff
                            new_Q[z, y, x, l] -= step_size * (np.sum(pairwise_diff, axis=(0, 1, 2)))

                        # apply opposite pairwise energy penalty to background label just to avoid background inpainting/dominance
                        new_Q[z, y, x, 0] += step_size * 0.5 * np.sum(pairwise_diff, axis=(0, 1, 2))

                    # Entropy change post-unary and pairwise update along label dimention
                    current_entropy = compute_entropy(np.exp(-new_Q[z, y, x, :]))
                    entropy_change = current_entropy - orig_entropy[z, y, x]
                    new_Q[z, y, x, :] -= entropy_weight * entropy_change

        # Convert updated potentials back to beliefs
        Q = np.exp(-new_Q)
        Q /= np.sum(Q, axis=-1, keepdims=True)  # Normalize

    #return np.argmax(Q, axis=-1)
    return Q
