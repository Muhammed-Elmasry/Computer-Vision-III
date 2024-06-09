import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..data.unsupervised_segmentation import KMeans


class InitializationShapeTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_features = 5
        self.num_samples = 10

        self.data = torch.rand(self.num_samples, self.num_features)

        self.centroids = None

    def test(self):
        kmeans = KMeans(self.num_clusters)

        self.centroids = kmeans.initialization(self.data)

        return self.centroids.shape == (self.num_clusters, self.num_features)

    def define_success_message(self):
        return (
            f"Congratulations: The initialization function returns the correct shape."
        )

    def define_failure_message(self):
        return f"The initialization function returns the incorrect shape. Expected {(self.num_clusters, self.num_features)}, got {self.centroids.shape}."


class InitCentroidsInDataTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_features = 5
        self.num_samples = 10

        self.data = torch.rand(self.num_samples, self.num_features)

        self.centroids = None

    def test(self):
        kmeans = KMeans(self.num_clusters)

        self.centroids = kmeans.initialization(self.data)
        for centroid in self.centroids:
            if centroid not in self.data:
                return False
        return True

    def define_success_message(self):
        return f"Congratulations: The initializations are found in the data."

    def define_failure_message(self):
        return f"At least one of the initializations is not found in the data."


class InitializationTest(MethodTest):
    def define_tests(self):
        return [
            InitializationShapeTest(),
            InitCentroidsInDataTest(),
        ]

    def define_method_name(self):
        return "initialization"


class DistanceToCentersShapeTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_features = 5

        self.num_datapoints = 10

        self.centroids = torch.rand(self.num_clusters, self.num_features)

        self.data = torch.rand(self.num_datapoints, self.num_features)

    def test(self):
        return True

    def define_success_message(self):
        return f"Congratulations: The output is correct"

    def define_failure_message(self):
        return f"The output is incorrect."


class DistanceToCentersOutputTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_features = 5

        self.num_datapoints = 10

        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.centroids = torch.load("exercise_code/test/dtc_centroids.pt")
        self.data = torch.load("exercise_code/test/dtc_data.pt")
        self.distances = torch.load("exercise_code/test/dtc_distances.pt")

    def test(self):
        kmeans = KMeans(self.num_clusters)
        self.output = kmeans.distance_to_centroids(self.centroids, self.data)

        return torch.all(torch.isclose(self.output, self.distances))

    def define_success_message(self):
        return f"Congratulations: The distance to the center function returns the correct distance."

    def define_failure_message(self):
        return f"The distance_to_center function returns the incorrect distance. Expected {self.distances}, got {self.output}"


class DistanceToCentersTest(MethodTest):
    def define_tests(self):
        return [
            DistanceToCentersShapeTest(),
            DistanceToCentersOutputTest(),
        ]

    def define_method_name(self):
        return "distance_to_centers"


class AssignClusterShapeTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_datapoints = 10

        self.distances = torch.rand(self.num_datapoints, self.num_clusters)

    def test(self):
        kmeans = KMeans(self.num_clusters)
        self.output = kmeans.assign_cluster(self.distances)

        return self.output.shape == (self.num_datapoints,)

    def define_success_message(self):
        return f"Congratulations: The output shape of assing_cluster is correct."

    def define_failure_message(self):
        return f"The output shape of assing_cluster is incorrect. Expected {(self.num_datapoints,)}, got {self.output.shape}"


class AssignClusterOutputTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_datapoints = 10

        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.distances = torch.load("exercise_code/test/ac_distances.pt")
        self.assignments = torch.load("exercise_code/test/ac_assignments.pt")

    def test(self):
        kmeans = KMeans(self.num_clusters)
        self.output = kmeans.assign_cluster(self.distances)

        return torch.all(torch.isclose(self.output, self.assignments))

    def define_success_message(self):
        return f"Congratulations: The output of assing_cluster is correct."

    def define_failure_message(self):
        return f"The output of assing_cluster is incorrect. Expected {self.assignments}, got {self.output}"


class AssignClusterTest(MethodTest):
    def define_tests(self):
        return [
            AssignClusterShapeTest(),
            AssignClusterOutputTest(),
        ]

    def define_method_name(self):
        return "assign_clusters"


class CalculateCentroidsShapeTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_features = 5
        self.num_datapoints = 10

        self.data = torch.rand(self.num_datapoints, self.num_features)
        self.assignments = torch.randint(
            low=0, high=self.num_clusters, size=(self.num_datapoints,)
        )

    def test(self):
        kmeans = KMeans(self.num_clusters)
        self.output = kmeans.calculate_centroids(
            self.data, self.assignments, self.num_clusters
        )

        return self.output.shape == (self.num_clusters, self.num_features)

    def define_success_message(self):
        return f"Congratulations: The output shape of calculate_centroids is correct."

    def define_failure_message(self):
        return f"The output shape of calculate_centroids is incorrect. Expected {(self.num_clusters, self.num_features)}, got {self.output.shape}"


class CalculateCentroidsOutputTest(UnitTest):
    def __init__(self):
        self.num_clusters = 3
        self.num_features = 5
        self.num_datapoints = 10

        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.data = torch.load("exercise_code/test/cc_data.pt")
        self.centroids = torch.load("exercise_code/test/cc_centroids.pt")
        self.assignments = torch.load("exercise_code/test/cc_assignments.pt")

    def test(self):
        kmeans = KMeans(self.num_clusters)
        self.output = kmeans.calculate_centroids(
            self.data, self.assignments, self.num_clusters
        )

        return torch.all(torch.isclose(self.output, self.centroids))

    def define_success_message(self):
        return f"Congratulations: The output of calculate_centroids is correct."

    def define_failure_message(self):
        return f"The output of calculate_centroids is incorrect. Expected {self.centroids}, got {self.output}"


class CalculateCentroidsTest(MethodTest):
    def define_tests(self):
        return [
            CalculateCentroidsShapeTest(),
            CalculateCentroidsOutputTest(),
        ]

    def define_method_name(self):
        return "calculate_centroids"


class KMeansTest(ClassTest):
    def define_tests(self):
        return [
            InitializationTest(),
            DistanceToCentersTest(),
            AssignClusterTest(),
            CalculateCentroidsTest(),
        ]

    def define_class_name(self):
        return "KMeans"


def test_k_means():
    test = KMeansTest()
    return test_results_to_score(test())
