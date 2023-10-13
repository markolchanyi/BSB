import numpy as np

def compute_local_pairwise_potentials(sigma_alpha, radius):
    """Computes local pairwise potentials for a given radius based on the distance."""
    z, y, x = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    squared_distance = x**2 + y**2 + z**2
    return np.exp(-squared_distance / (2 * sigma_alpha**2))

def dense_crf_inference(cnn_output, iterations=4, sigma_alpha=0.5, radius=7):
    depth, height, width, num_labels = cnn_output.shape
    unary = -np.log(cnn_output + 1e-10)*6  # Added a small constant to avoid log(0)

    pairwise_full = compute_local_pairwise_potentials(sigma_alpha, radius)

    Q = np.copy(cnn_output)  # Start with the CNN output as the initial beliefs

    for _ in range(iterations):
        new_Q = np.copy(unary)  # Initialize with unary potentials
        for z in range(10, depth - 10):
            for y in range(10, height - 10):
                for x in range(10, width - 10):
                    local_start = [max(0, z - radius), max(0, y - radius), max(0, x - radius)]
                    local_end = [min(depth, z + radius + 1), min(height, y + radius + 1), min(width, x + radius + 1)]
                    local_slice = tuple(slice(s, e) for s, e in zip(local_start, local_end))
                    local_region = Q[local_slice]
                    
                    # Offset from the center
                    pairwise_start = [radius + s - z for s, z in zip(local_start, [z, y, x])]
                    pairwise_end = [start + e - s for start, s, e in zip(pairwise_start, local_start, local_end)]
                    pairwise_slice = tuple(slice(s, e) for s, e in zip(pairwise_start, pairwise_end))
                    local_pairwise = pairwise_full[pairwise_slice]

                    # Adjust the pairwise potentials shape to match local_region spatial dimensions
                    reshaped_pairwise = local_pairwise[..., np.newaxis]
                    
                    # Calculate message from pairwise potentials
                    pairwise_message = np.sum(reshaped_pairwise * local_region, axis=(0, 1, 2))

                    # Update beliefs using pairwise message
                    new_Q[z, y, x, :] -= pairwise_message

        # Convert updated potentials back to beliefs
        Q = np.exp(-new_Q)
        Q /= np.sum(Q, axis=-1, keepdims=True)  # Normalize

    return np.argmax(Q, axis=-1)

