import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh

import torch.nn.functional as F


# https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to HSV space. The RGB image should have value between 0 and 1

    Args:
        rgb (torch.Tensor): RGB image of shape (B, 3, N, M)

    Returns:
        torch.Tensor: HSV image of shape (B, 3, N, M)
    """
    cmax, cmax_idx = torch.max(rgb, dim=-3, keepdim=True)
    cmin = torch.min(rgb, dim=-3, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[..., 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[..., 1:2, :, :] - rgb[..., 2:3, :, :]) / delta) % 6)[
        cmax_idx == 0
    ]
    hsv_h[cmax_idx == 1] = (((rgb[..., 2:3, :, :] - rgb[..., 0:1, :, :]) / delta) + 2)[
        cmax_idx == 1
    ]
    hsv_h[cmax_idx == 2] = (((rgb[..., 0:1, :, :] - rgb[..., 1:2, :, :]) / delta) + 4)[
        cmax_idx == 2
    ]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=-3)


def calc_W_knn(image: torch.Tensor, n_neighbors: int) -> torch.Tensor:
    # image C, H, W
    hsv_image = rgb2hsv_torch(image)  # image 3, H, W
    h, w = image.shape[-2:]

    ########################################################################
    # TODO:                                                                #
    # Calculate the image feature that are used to calcualte the knn       #
    # affinity matrix for spectral clustering. Features should have shape  #
    # H*W x 6                                                              #
    #                                                                      #
    # features = ...                                                       #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    W_knn = torch.zeros(features.shape[0], features.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(
        features
    )
    distances, indices = nbrs.kneighbors(features)
    distances = torch.from_numpy(distances[:, 1:]).flatten(0, 1).to(W_knn.dtype)
    y_indices = torch.from_numpy(indices[:, 1:]).flatten(0, 1).tolist()
    x_indices = (
        torch.arange(0, features.shape[0], 1)[:, None]
        .repeat(1, n_neighbors)
        .flatten(0, 1)
        .tolist()
    )

    W_knn[y_indices, x_indices] = distances

    return W_knn


def calc_W_feat(feature_map: torch.Tensor) -> torch.Tensor:
    """Calculates an affinity matrix based on features.

    Args:
        feature_map (torch.Tensor): 2-D feature tensor (N, M) of N datapoints of size M

    Returns:
        torch.Tensor: affinty matrix (N, N)
    """

    ########################################################################
    # TODO:                                                                #
    # Calculate the fearture based affinity matrix for spectral            #
    # clustering. The affinity matrix should be of size N x N and have     #
    # normalized non-negative entries between 0 and 1.Features that are    #
    # closer should have a higher affinity.                                #
    #                                                                      #
    # W_feat = ...                                                         #
    ########################################################################


    pass

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return W_feat


def downscale(image: torch.Tensor, scale_factor: int = 8) -> torch.Tensor:
    pooling = torch.nn.AvgPool2d(scale_factor)
    return pooling(image)


# Adapted from https://github.com/lukemelas/deep-spectral-segmentation
class SpectralClustering:
    def __init__(
        self,
        scaling: int,
        max_clusters: int,
        max_eigenvecs: int,
    ) -> None:
        self.scaling = scaling
        self.max_clusters = max_clusters
        self.max_eigenvecs = max_eigenvecs

    def original_cluster(
        self,
        img: torch.Tensor,
        feats: torch.Tensor,
        image_color_lambda: float = 10.0,
        normalized: bool = True,
        upscale: bool = True,
    ) -> torch.Tensor:
        C, H, W = img.shape
        H_patch, W_patch = H // self.scaling, W // self.scaling

        if normalized:
            feats = F.normalize(feats, p=2, dim=-1)

        ### Feature affinities
        W_feat = calc_W_feat(feats.flatten(0, 1)).detach().cpu().numpy()

        ### Color affinities
        # If we are fusing with color affinites, then load the image and compute
        if image_color_lambda > 0:
            # Color affinities (of type scipy.sparse.csr_matrix)
            W_knn = calc_W_knn(downscale(img, self.scaling), 20).detach().cpu().numpy()
        else:
            # No color affinity
            W_knn = 0

        # Combine
        W_comb = W_feat + W_knn * image_color_lambda  # combination
        D_comb = np.diag(
            np.sum(W_comb, axis=1)
        )  # is dense or sparse faster? not sure, should check

        # Extract eigenvectors
        K = 20
        try:
            eigenvalues, eigenvectors = eigsh(
                D_comb - W_comb, k=K, sigma=0, which="LM", M=D_comb
            )
        except:
            eigenvalues, eigenvectors = eigsh(
                D_comb - W_comb, k=K, which="SM", M=D_comb
            )
        eigenvalues, eigenvectors = (
            torch.from_numpy(eigenvalues),
            torch.from_numpy(eigenvectors.T).float(),
        )

        threshold = 0.0
        segmap = (eigenvectors[1].cpu().numpy() > threshold).reshape(H_patch, W_patch)

        if upscale:
            upscaled_cl = np.repeat(
                np.repeat(segmap, self.scaling, axis=0),
                self.scaling,
                axis=1,
            )
            return torch.from_numpy(upscaled_cl)

        return torch.from_numpy(segmap)
