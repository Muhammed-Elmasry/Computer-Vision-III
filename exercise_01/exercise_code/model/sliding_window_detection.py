from math import floor
from typing import Tuple
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from exercise_code.model.hog import HoG


def sliding_window_detection(
    image: torch.Tensor, model: nn.Module, patch_size: Tuple[int, int], stride: int = 1
) -> torch.Tensor:
    grayscale_image = transforms.functional.rgb_to_grayscale(image).float()
    block_size = 8
    num_bins = 9
    H_p, W_p = patch_size[0], patch_size[1]
    ########################################################################
    # TODO:                                                                #
    # Perform person detection in a sliding window manner with HoG         #
    # features and a trained classification model. For this                #
    #                                                                      #
    # - get patches of the grayscale image by using the patch_size and     #
    # stride parameters. Be aware of the stride parameter and don't pad    #
    # the image.                                                           #
    #                                                                      #
    # - compute the HoG features for each patch. The function call is      #
    # hog_features = HoG(patches, block_size, num_bins), where             #
    # block_size and num_bins are hard coded to 8 and 9, respectively.     #
    # Use the HoG function on the batch of grey valued patches of shape    #
    # (B, patch_size[0], patch_size[1]) to compute the flattened HoG       #
    # features.                                                            #
    #                                                                      #
    # - use the model for prediction.                                      #
    #                                                                      #
    # - reshape the output of the model to have 2-spatial dimensions       #
    # and classification scores as values.                                 #
    #                                                                      #
    # Input:                                                               #
    # A single input image of size (3, H, W).                              #
    # The stride and patch_size for of the sliding window operation.       #
    # (With the provided parameters, the hog_features fit                  #
    # the expected input size of the classifier)                           #
    # The classification model, which takes a batch of flattened feature   #
    # patches (hog_features is already flattened)                          #
    # Output:                                                              #
    # An collection if classified patches. The ouput shape should be       #
    # (floor((H-(patch_size[0]-1)-1)/stride+1),                            #
    # floor((W-(patch_size[1]-1)-1)/stride+1)).                            #
    #                                                                      #
    ########################################################################

    _, H, W = grayscale_image.shape

    # Calculate the number of patches in each dimension
    num_patches_H = (H - H_p) // stride + 1
    num_patches_W = (W - W_p) // stride + 1

    # Initialize the output detection image
    detection_image = torch.zeros(num_patches_H, num_patches_W)

    # Loop through patches and perform detection
    for i in range(0, H - H_p + 1, stride):
        for j in range(0, W - W_p + 1, stride):
            # Extract patch from the image
            patch = grayscale_image[:, i:i + H_p, j:j + W_p]

            # Compute HoG features for the patch
            hog_features = HoG(patch, block_size, num_bins)

            # Flatten HoG features
            hog_features = hog_features.view(1, -1)

            # Use the model for prediction
            predictions = model(hog_features)

            # Reshape the output to have 2-spatial dimensions
            predictions = predictions.view(-1)

            # Assign the prediction to the corresponding position in the detection image
            detection_image[i // stride, j // stride] = predictions.item()

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return detection_image
