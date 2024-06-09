import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..data.unsupervised_segmentation.spectral_clustering import calc_W_feat, calc_W_knn


class WFeatShapeTest(UnitTest):
    def __init__(self):
        self.H = 48
        self.W = 64
        self.channels = 32

    def test(self):
        self.W_feat = calc_W_feat(torch.rand(self.H * self.W, self.channels))
        return self.W_feat.shape == (self.H * self.W, self.H * self.W)

    def define_success_message(self):
        return f"Congratulations: The feature affinity matrix has the correct shape."

    def define_failure_message(self):
        return f"The feature affinity matrix does not have the correct shape. Expected {(self.H * self.W, self.H * self.W)}, got {self.W_feat.shape}."


class WFeatOutputTest(UnitTest):
    def __init__(self):
        self.H = 4
        self.W = 8
        self.channels = 8
        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.features = torch.load("exercise_code/test/w_feat_features.pt")
        self.output = torch.load("exercise_code/test/w_feat_output.pt")

    def test(self):
        self.W_feat = calc_W_feat(self.features)
        return torch.all(torch.isclose(self.W_feat, self.output))

    def define_success_message(self):
        return f"Congratulations: The feature affinity matrix was calculated correctly."

    def define_failure_message(self):
        return f"The feature affinity matrix was not calculated correctly. Expected {self.output}, got {self.W_feat}."


class WFeatTest(MethodTest):
    def define_tests(self):
        return [
            WFeatShapeTest(),
            WFeatOutputTest(),
        ]

    def define_method_name(self):
        return "calc_w_feat"


class WKNNShapeTest(UnitTest):
    def __init__(self):
        self.H = 48
        self.W = 64
        self.channels = 3
        self.num_neigbors = 3

    def test(self):
        self.W_knn = calc_W_knn(
            torch.rand(self.channels, self.H, self.W), self.num_neigbors
        )
        return self.W_knn.shape == (self.H * self.W, self.H * self.W)

    def define_success_message(self):
        return f"Congratulations: The knn affinity matrix has the correct shape."

    def define_failure_message(self):
        return f"The knn affinity matrix does not have the correct shape. Expected {(self.H * self.W, self.H * self.W)}, got {self.W_knn.shape}."


class WKNNOutputTest(UnitTest):
    def __init__(self):
        self.H = 4
        self.W = 8
        self.channels = 3
        self.num_neigbors = 3
        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.image = torch.load("exercise_code/test/w_knn_image.pt")
        self.output = torch.load("exercise_code/test/w_knn_output.pt")

    def test(self):
        self.W_knn = calc_W_knn(self.image, self.num_neigbors)
        return torch.all(torch.isclose(self.W_knn, self.output))

    def define_success_message(self):
        return f"Congratulations: The knn affinity matrix was calculated correctly."

    def define_failure_message(self):
        return f"The knn affinity matrix was not calculated correctly. Expected {self.output}, got {self.W_knn}."


class WKNNTest(MethodTest):
    def define_tests(self):
        return [
            WKNNShapeTest(),
            WKNNOutputTest(),
        ]

    def define_method_name(self):
        return "calc_w_knn"


class SpectralClustering(ClassTest):
    def define_tests(self):
        return [
            WFeatTest(),
            WKNNTest(),
        ]

    def define_class_name(self):
        return "SpectralClustering"


def test_spectral_clustering():
    test = SpectralClustering()
    return test_results_to_score(test())
