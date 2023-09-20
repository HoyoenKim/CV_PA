import numpy as np
from tqdm import tqdm


def aggregate_cost_volume(cost_volume):
    height, width, max_disparity = cost_volume.shape
    aggregated_costs = np.zeros_like(cost_volume, dtype=np.float32)
 
    forward_pass = [(1, 0), (1, 1), (0, 1), (-1, 1)]  # Dy, Dx pairs for forward pass
    backward_pass = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]  # Dy, Dx pairs for backward pass

    for idx, (dy, dx) in enumerate(forward_pass):
        # Shift the cost volume by (dy, dx) and pad the shifted values
        shifted_costs = np.roll(cost_volume, (dy, dx), axis=(0, 1))
        shifted_costs[:dy, :] = np.inf
        shifted_costs[:, :dx] = np.inf
        aggregated_costs += shifted_costs

    for idx, (dy, dx) in enumerate(backward_pass):
        # Shift the cost volume by (dy, dx) and pad the shifted values
        shifted_costs = np.roll(cost_volume, (dy, dx), axis=(0, 1))
        shifted_costs[dy:, :] = np.inf
        shifted_costs[:, dx:] = np.inf
        aggregated_costs += shifted_costs

    print(aggregated_costs.shape)
    aggregated_volume = np.sum(aggregated_costs, axis=3)
    return aggregated_volume
