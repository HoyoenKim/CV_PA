import numpy as np


def Birchfield_Tomasi_dissimilarity(left_image, right_image, d):
    # TODO: Implement Birchfield-Tomasi dissimilarity
    # Hint: Fill undefined elements with np.inf at the end

    raise NotImplementedError("Birchfield_Tomasi_dissimilarity function has not been implemented yet")

    left_cost_volume = None
    right_cost_volume = None

    left_disparity = left_cost_volume.argmin(axis=2)
    right_disparity = right_cost_volume.argmin(axis=2)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity

def Sum_Of_Absolute_Differences(left_image, right_image, max_disparity):
    # Get image dimensions
    height, width = left_image.shape

    # Initialize cost volumes for the left and right images
    left_cost_volume = np.zeros((height, width, max_disparity), dtype=np.float32)
    right_cost_volume = np.zeros((height, width, max_disparity), dtype=np.float32)

    # Iterate over disparity levels
    for d in range(max_disparity):
        # Calculate the absolute difference for the left image
        if d == 0:
            left_cost_volume[:, :, d] = np.abs(left_image - right_image)
        else:
            left_cost_volume[:, :-d, d] = np.abs(left_image[:, :-d] - right_image[:, d:])

        # Calculate the absolute difference for the right image
        if d == 0:
            right_cost_volume[:, :, d] = np.abs(right_image - left_image)
        else:
            right_cost_volume[:, d:, d] = np.abs(right_image[:, d:] - left_image[:, :-d])
    
    # Find the disparity with minimum cost for the left and right images
    left_disparity = np.argmin(left_cost_volume, axis=2)
    right_disparity = np.argmin(right_cost_volume, axis=2)

    return left_cost_volume, right_cost_volume, left_disparity, right_disparity
