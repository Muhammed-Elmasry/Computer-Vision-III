from typing import Dict
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# class KMeans(nn.Module):
#     def __init__(self, num_clusters: int, max_iter: int = 100) -> None:
#         super().__init__()
#         self.num_clusters = num_clusters
#         self.max_iter = max_iter
#         self.prev_assignments = None

#         self.centroids = None

#     def initialization(self, data: torch.Tensor) -> torch.Tensor:
#         """For each datapoint, calculate its distance to each cluster center.

#         Args:
#             data (torch.Tensor): 2-D feature tensor (N x M) of N datapoints of size M

#         Returns:
#             torch.Tensor: 2-D tensor (C, M) of C cluster centroids of size M
#         """

#         ########################################################################
#         # TODO:                                                                #
#         # Compute initial positons for the centroids based of kmeans++. The    #
#         # output should be a C x M dimensional tensor where C is given by      #
#         # self.num_clusters                                                    #
#         # NOTE: You can Categorical from torch.distributions.categorical       #
#         # to draw random samples from a categorical distribution. See          #
#         # https://pytorch.org/docs/stable/distributions.html#categorical       #
#         # for more details on how to use it                                    #
#         #                                                                      #
#         # centroids = ...                                                      #
#         ########################################################################


#         pass

#         ########################################################################
#         #                           END OF YOUR CODE                           #
#         ########################################################################

#         return centroids

#     def distance_to_centroids(self, data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
#         """For each datapoint, calculate its distance to each cluster centroids.

#         Args:
#             data (torch.Tensor): 2-D feature tensor (N x M) of N datapoints of size M
#             centroids (torch.Tensor): 2-D feature tensor (C x M) of C centroids of size M

#         Returns:
#             torch.Tensor: distance tensor (N x C) of the N datapoints to the K centroids
#         """
#         ########################################################################
#         # TODO:                                                                #
#         # Calculate the euclidean distance of all points to all centroids.     #
#         # The output should be a tensor of shape N x C                         #
#         #                                                                      #
#         # distances = ...                                                      #
#         ########################################################################


#         pass

#         ########################################################################
#         #                           END OF YOUR CODE                           #
#         ########################################################################

#         return distances

#     def assign_cluster(self, distances: torch.Tensor) -> torch.Tensor:
#         """Assign the cluster to each datapoint based on the distances to the centers.

#         Args:
#             distances (torch.Tensor): 2-D feature tensor (N x C) of distances of N datapoints to C cluster centroids

#         Returns:
#             torch.Tensor: cluster labels (N) giving the cluster with minimum distance for each of the N datapoints
#         """

#         ########################################################################
#         # TODO:                                                                #
#         # Assign each datapoint to a cluster based on the minimum distance.    #
#         # The output should be a N dimensional tensor with entries going from  #
#         # 0...self.num_clusters-1                                              #
#         #                                                                      #
#         # assignments = ...                                                    #
#         ########################################################################


#         pass

#         ########################################################################
#         #                           END OF YOUR CODE                           #
#         ########################################################################

#         return assignments

#     def calculate_centroids(self, data: torch.Tensor, assignments: torch.Tensor, num_clusters: int) -> torch.Tensor:
#         """Calculate the new centroids of the clusters based on the assignments

#         Args:
#             data (torch.Tensor): 2-D feature tensor (N x M) of N datapoints of size M
#             assignments (torch.Tensor): 1-D feature tensor (N) containing the cluster assignments for each datapoint
#             num_clusters (int): number on clusters

#         Returns:
#             torch.Tensor: new cluster center tensor (C x M)
#         """

#         ########################################################################
#         # TODO:                                                                #
#         # Calculate the new centroid poisitions based on which datapoints were #
#         # assigned to which cluster                                            #
#         # NOTE: Caluclation of the new optimal centroids is denpendent on the  #
#         # distance metric used for assignment. For the euclidean distance it   #
#         # is the mean value.                                                   #
#         #                                                                      #
#         # centroids = ...                                                      #
#         ########################################################################


#         pass

#         ########################################################################
#         #                           END OF YOUR CODE                           #
#         ########################################################################

#         return centroids


class KMeans(nn.Module):
    def __init__(self, num_clusters: int, max_iter: int = 100) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.prev_assignments = None

        self.centroids = None

    def initialization(self, data: torch.Tensor) -> torch.Tensor:
        N, M = data.shape
        centroids = torch.empty(self.num_clusters, M, device=data.device)

        # Step 1: Randomly select the first centroid from the data points
        first_centroid_idx = torch.randint(0, N, (1,)).item()
        centroids[0] = data[first_centroid_idx]

        for i in range(1, self.num_clusters):
            # Step 2: Compute distances from the closest centroid for all points
            distances = self.distance_to_centroids(data, centroids[:i]).min(dim=1)[0]

            # Step 3: Choose new centroid with probability proportional to the squared distance
            probs = distances / distances.sum()
            categorical = Categorical(probs)
            centroid_idx = categorical.sample().item()

            centroids[i] = data[centroid_idx]

        return centroids

    def distance_to_centroids(self, data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(data, centroids, p=2)
        return distances

    def assign_cluster(self, distances: torch.Tensor) -> torch.Tensor:
        assignments = distances.argmin(dim=1)
        return assignments

    def calculate_centroids(self, data: torch.Tensor, assignments: torch.Tensor, num_clusters: int) -> torch.Tensor:
        N, M = data.shape
        centroids = torch.zeros(num_clusters, M, device=data.device)

        for i in range(num_clusters):
            assigned_data = data[assignments == i]
            if len(assigned_data) > 0:
                centroids[i] = assigned_data.mean(dim=0)

        return centroids

    def is_converged(self, assignments: torch.Tensor) -> bool:
        """Caluclates if the k-means algorithm has converged. Convergences happend if the assignments didn't change since the last iteration.

        Args:
            assignments (torch.Tensor): 1-D feature tensor (N) containing the cluster assignments for each datapoint

        Returns:
            bool: True if assigments didn't change, False else
        """
        if self.prev_assignments is None:
            self.prev_assignments = assignments
            return False

        is_converged = torch.sum(torch.abs(self.prev_assignments - assignments)).item() == 0
        self.prev_assignments = assignments
        return is_converged

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        distances = self.distance_to_centroids(data, self.centroids)
        assignments = self.assign_cluster(distances)

        return assignments

    # def train(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
    def train(self, data: torch.Tensor) -> None:
        self.centroids = self.initialization(data)

        distances = self.distance_to_centroids(data, self.centroids)
        assignments = self.assign_cluster(distances)

        epoch_loss = torch.min(distances, dim=-1)[0].mean()

        for idx in range(1, self.max_iter):
            distances = self.distance_to_centroids(data, self.centroids)
            epoch_loss = torch.min(distances, dim=-1)[0].mean()

            assignments = self.assign_cluster(distances)
            self.centroids = self.calculate_centroids(data, assignments, self.num_clusters)

            if idx % 10 == 0:
                print(f"{f'Epoch {idx} Loss:' : >20} {epoch_loss}")

            if self.is_converged(assignments):
                break

        # return {
        #     "centroids": self.centroids,
        #     "assignments": assignments,
        # }
