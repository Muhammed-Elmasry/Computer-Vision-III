# import torch
# import torch.nn as nn


# class LinearProbingNet(nn.Module):
#     def __init__(self, input_dim: int) -> None:
#         super().__init__()
#         ########################################################################
#         # TODO:                                                                #
#         # Define a ONE layer neural network with a 1x1 convolution and a       #
#         # sigmoid non-linearity to do binary classification on an image        #
#         # NOTE: the network receives a batch of feature maps of shape          #
#         # B x H x W x input_dim and should output a binary classification of   #
#         # shape B x H x W                                                      #
#         ########################################################################


#         pass

#         ########################################################################
#         #                           END OF YOUR CODE                           #
#         ########################################################################

#     def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
#         # x is a batch of feature maps of shape B x H x W x feat_dim

#         ########################################################################
#         # TODO:                                                                #
#         # Do the forward pass of you defined network                           #
#         # prediction = ...                                                     #
#         ########################################################################


#         pass

#         ########################################################################
#         #                           END OF YOUR CODE                           #
#         ########################################################################

#         return prediction

import torch
import torch.nn as nn

class LinearProbingNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Define a 1x1 convolutional layer followed by a sigmoid activation function
        # The convolution layer changes the channel dimension from input_dim to 1
        self.conv = nn.Conv2d(input_dim, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Check if the input tensor is in the correct shape (B x input_dim x H x W)
        if x.shape[1] != self.conv.in_channels:
            # Permute dimensions to bring the input_dim to the second dimension (channel)
            x = x.permute(0, 3, 1, 2)

        # Apply the convolution followed by the sigmoid activation
        x = self.conv(x)
        x = self.sigmoid(x)

        # No need to squeeze the channel dimension; return the tensor as is
        return x

