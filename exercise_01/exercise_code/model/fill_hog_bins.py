import torch


def fill_hog_bins(gradient_norm: torch.Tensor, gradient_angle: torch.Tensor, num_bins: int) -> torch.Tensor:
    assert gradient_norm.shape == gradient_angle.shape
    device = gradient_norm.device

    ########################################################################
    # TODO:                                                                #
    # Based on the given gradient norm and angle, fill the Histogram of    #
    # Orientatied Gradients bins.                                          #
    # For this, first determine the two bins a gradient should be part of  #
    # based on the gradient angle. Then, based on the distance to the bins #
    # fill the bins with a weighting of the gradient norm.                 #
    # Input:                                                               #
    # Both gradient_norm and gradient_angle have the shape (B, N), where   #
    # N is the flatten cell.                                               #
    # The angle is given in degrees with values in the range [0.0, 180.0). #
    # (the angles 0.0 and 180.0 are equivalent.)                           #
    # Output:                                                              #
    # The output is a histogram over the flattened cell with num_bins      #
    # quantized values and should have the shape (B, num_bins)             #
    #                                                                      #
    # NOTE: Keep in mind the cyclical nature of the gradient angle and     #
    # its effects on the bins.                                             #
    # NOTE: make sure, the histogram_of_oriented_gradients are on the same #
    # device as the gradient inputs. In general be mindful of the device   #
    # of the tensors.                                                      #
    # histogram_of_oriented_gradients = ...                                #
    ########################################################################

    # Determine bin positions
    bin_width = 180.0 / num_bins
    bin_positions = torch.arange(0, 180, bin_width, device=device)

    # Find the nearest bin for each angle
    bin1 = ((gradient_angle / bin_width).long() % num_bins)
    bin2 = (bin1 + 1) % num_bins

    # Calculate the difference between angles and bins
    diff_angle_bin1 = torch.abs(gradient_angle - bin_positions[bin1])
    diff_angle_bin2 = bin_width - diff_angle_bin1

    # Compute weights for each bin based on the differences
    weight_bin1 = diff_angle_bin2 / bin_width
    weight_bin2 = diff_angle_bin1 / bin_width

    # Fill the histogram of oriented gradients based on weights and gradient norms
    histogram_of_oriented_gradients = torch.zeros((gradient_norm.shape[0], num_bins), device=device)
    for i in range(gradient_norm.shape[0]):
        histogram_of_oriented_gradients[i].index_add_(0, bin1[i], weight_bin1[i] * gradient_norm[i])
        histogram_of_oriented_gradients[i].index_add_(0, bin2[i], weight_bin2[i] * gradient_norm[i])


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return histogram_of_oriented_gradients
