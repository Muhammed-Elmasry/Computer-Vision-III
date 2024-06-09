from typing import Tuple
import torch.nn as nn
import torch


def compute_image_gradient(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # images B x H x W

    ########################################################################
    # TODO:                                                                #
    # Compute the 2-dimenational gradient for a given grey image of size   #
    # B x H x W. The return values of this function should be the norm and #
    # the angle of this gradient vector.                                   #
    # NOTE: first, calculate the gradient in x and y direction             #
    # (you will need add padding to the image boundaries),                 #
    # then, compute the vector norm and angle.                             #
    # The angle of a given gradient angle is defined                       #
    # in degrees (range=0.,,,.360).                                        #
    # NOTE: The angle is defined counter-clockwise angle between the       #
    # gradient and the unit vector along the x-axis received from atan2.   #
    ########################################################################
    # gradient_x = torch.zeros_like(images)
    # gradient_y = torch.zeros_like(images)

    # for i in range(images.shape[0]):  # Iterate over the batch dimension
    #     for x in range(images.shape[1]):  # Height dimension
    #         for y in range(images.shape[2]):  # Width dimension
    #             # Compute gradients in x-direction (central differences)
    #             if x - 1 >= 0 and x + 1 < images.shape[1]:
    #                 gradient_x[i, x, y] = (images[i, x + 1, y] - images[i, x - 1, y])

    #             # Compute gradients in y-direction (central differences)
    #             if y - 1 >= 0 and y + 1 < images.shape[2]:
    #                 gradient_y[i, x, y] = (images[i, x, y + 1] - images[i, x, y - 1])

    gradient_x = torch.zeros_like(images)
    gradient_x[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]

    # Compute gradients in y-direction (central differences)
    gradient_y = torch.zeros_like(images)
    gradient_y[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]

    # Compute the norm of the gradient
    norm = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Compute the angle of the gradient using arctan2 and convert to degrees
    angle = torch.atan2(gradient_y, gradient_x)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return norm, angle
