from typing import Dict, Tuple
import torch
import torch.nn as nn

from exercise_code.data.unsupervised_segmentation.k_means import KMeans


class GaussianMixtureModels:
    def __init__(
        self,
        num_classes: int,
        kmeans: KMeans,
        max_iter: int = 100,
        full_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_iter = max_iter

        self.kmeans = kmeans
        self.kmeans.num_clusters = num_classes
        self.full_init = full_init

        self.mixing_coefficients = None
        self.centroids = None
        self.covariances = None
        self.probabilties = None
        self.responsibilities = None

        self.prev_expected_value = None
        self.eps = 1.0e-10

    def initialization(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        covariances = torch.eye(data.shape[-1])[None].repeat(self.num_classes, 1, 1)
        mixing_coefficients = torch.ones(self.num_classes) / self.num_classes
        if self.full_init:
            centroids = self.kmeans(data)["centroids"]
        else:
            centroids = self.kmeans.initialization(data)

        return centroids, covariances, mixing_coefficients

    def e_step(
        self,
        data: torch.Tensor,
        mixing_coefficients: torch.Tensor,
        centroids: torch.Tensor,
        covariances: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate the new gaussian parameters of the clusters based on the responsibility

        Args:
            data (torch.Tensor): 2-D feature tensor (N x M) of N datapoints of size M
            mixing_coefficients (torch.Tensor): 1-D feature tensor (C) containing likelihood for the classes
            centroids (torch.Tensor): 2-D feature tensor (C, M) of the centroid positions for each gaussian model
            covariances (torch.Tensor): 3-D feature tensor (C, M, M) of the covariance matrices for each gaussian model

        Returns:
            torch.Tensor: responsibility (C, N)
            torch.Tensor: probabilities (C, N)
        """

        ########################################################################
        # TODO:                                                                #
        # Do the e-Step of the EM alogrithm for the GMM models.                #
        # Caclulate the probailities of shape C x N of each data point         #
        # belonging to each cluster based on its gaussian parameters. From     #
        # these probabilties calculate the responsibilies of shape C x N of    #
        # the gaussian mixture model.                                          #
        #                                                                      #
        # probabilities = ...                                                  #
        # responsibility = ...                                                 #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return responsibility, probabilities

    def m_step(
        self, data: torch.Tensor, responsibility: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the new gaussian parameters of the clusters based on the responsibility

        Args:
            data (torch.Tensor): 2-D feature tensor (N x M) of N datapoints of size M
            responsibility (torch.Tensor): 2-D feature tensor (C, N) containing responsibilities for each class

        Returns:
        TODO: fix output type
            torch.Tensor: new cluster center tensor (C, M)
        """

        ########################################################################
        # TODO:                                                                #
        # Do the m-Step of the EM alogrithm for the GMM models.                #
        # Recalulate the parameters of the gaussian mixture model.             #
        #                                                                      #
        # mixing_coeffients = ...                                              #
        # centroids = ...                                                      #
        # covariances = ...                                                    #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return centroids, covariances, mixing_coefficients

    def calculate_expected_value(self, probabilities: torch.Tensor) -> float:
        return torch.mean(torch.log(torch.sum(probabilities, dim=0))).item()

    def is_converged(self, probabilities: torch.Tensor) -> bool:
        expected_value = self.calculate_expected_value(probabilities)

        if self.prev_expected_value is None:
            self.prev_expected_value = expected_value
            return False

        is_converged = abs(expected_value - self.prev_expected_value) < 1.0e-3
        self.prev_expected_value = expected_value
        return is_converged

    def train(self, data: torch.Tensor) -> None:
        (
            self.centroids,
            self.covariances,
            self.mixing_coefficients,
        ) = self.initialization(data)

        for idx in range(self.max_iter):
            self.responsibilities, self.probabilities = self.e_step(
                data, self.mixing_coefficients, self.centroids, self.covariances
            )
            self.centroids, self.covariances, self.mixing_coefficients = self.m_step(
                data, self.responsibilities
            )

            if self.is_converged(self.probabilities):
                print(f"Converged after {idx + 1} iterations")
                break
        # self.responsibilities, _ = self.e_step(data, self.mixing_coefficients, self.centroids, self.covariances)

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        responsibilities, _ = self.e_step(
            data, self.mixing_coefficients, self.centroids, self.covariances
        )
        return responsibilities
