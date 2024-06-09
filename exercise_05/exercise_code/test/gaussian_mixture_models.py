import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..data.unsupervised_segmentation import GaussianMixtureModels, KMeans


def generate_rand_centroids(num_clusters, data_dim):
    centroids = torch.rand(num_clusters, data_dim)
    return centroids


def generate_rand_covariances(num_clusters, data_dim):
    covariances = torch.rand(num_clusters, data_dim, data_dim)
    covariances = covariances @ covariances.transpose(-1, -2)
    return covariances


def generate_rand_mixing_coefficients(num_clusters):
    mixing_coefficients = torch.rand(num_clusters)
    mixing_coefficients /= mixing_coefficients.sum()
    return mixing_coefficients


def generate_rand_responsibilities(num_clusters, num_datapoints):
    responsibilities = torch.rand(num_clusters, num_datapoints)
    responsibilities /= responsibilities.sum(dim=0, keepdim=True)
    return responsibilities


class EStepShapeTest(UnitTest):
    def __init__(self):
        self.num_clusters = 2
        self.max_iter = 10
        self.num_datapoints = 5
        self.data_dim = 3

    def test(self):
        data = torch.rand(self.num_datapoints, self.data_dim)
        kmeans = KMeans(self.num_clusters, self.max_iter)
        gmm = GaussianMixtureModels(
            self.num_clusters, kmeans, self.max_iter, full_init=False
        )

        self.responsibilities, self.probabilities = gmm.e_step(
            data,
            generate_rand_mixing_coefficients(self.num_clusters),
            generate_rand_centroids(self.num_clusters, self.data_dim),
            generate_rand_covariances(self.num_clusters, self.data_dim),
        )
        return self.responsibilities.shape == (
            self.num_clusters,
            self.num_datapoints,
        ) and self.probabilities.shape == (self.num_clusters, self.num_datapoints)

    def define_success_message(self):
        return f"Congratulations: The e-step produces the correct shape for the responsibilities and the probabilities."

    def define_failure_message(self):
        error_msg = ""
        if self.responsibilities.shape != (self.num_clusters, self.num_datapoints):
            error_msg += f"The responsibilities have the wrong shape. Expected {(self.num_clusters, self.num_datapoints)}, got {self.responsibilities.shape}."
        if self.probabilities.shape != (self.num_clusters, self.num_datapoints):
            error_msg += f"The probabilities have the wrong shape. Expected {(self.num_clusters, self.num_datapoints)}, got {self.probabilities.shape}."
        return f"The e-step does not produce the correct shape. {error_msg}"


class EStepOutputTest(UnitTest):
    def __init__(self):
        self.num_clusters = 2
        self.max_iter = 10
        self.num_datapoints = 5
        self.data_dim = 3
        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        self.data = torch.load("exercise_code/test/e_step_data.pt")
        self.centroids = torch.load("exercise_code/test/e_step_centroids.pt")
        self.covariances = torch.load("exercise_code/test/e_step_covariances.pt")
        self.mixing_coefficients = torch.load(
            "exercise_code/test/e_step_mixing_coefficients.pt"
        )
        self.responsibilities = torch.load(
            "exercise_code/test/e_step_responsibilities.pt"
        )
        self.probabilities = torch.load("exercise_code/test/e_step_probabilities.pt")

    def test(self):
        kmeans = KMeans(self.num_clusters, self.max_iter)
        gmm = GaussianMixtureModels(
            self.num_clusters, kmeans, self.max_iter, full_init=False
        )

        self.out_responsibilities, self.out_probabilities = gmm.e_step(
            self.data,
            self.mixing_coefficients,
            self.centroids,
            self.covariances,
        )
        return torch.all(
            torch.isclose(self.out_responsibilities, self.responsibilities)
        ) and torch.all(torch.isclose(self.out_probabilities, self.probabilities))

    def define_success_message(self):
        return f"Congratulations: The e-step calculates the probabilities and responsibilities correctly."

    def define_failure_message(self):
        error_msg = ""
        if not torch.all(
            torch.isclose(self.out_responsibilities, self.responsibilities)
        ):
            error_msg += f"The responsibilities were not calculated correctly. Expected {self.responsibilities}, got {self.out_responsibilities}."
        if not torch.all(torch.isclose(self.out_probabilities, self.probabilities)):
            error_msg += f"The probabilities were not calculated correctly. Expected {self.probabilities}, got {self.out_probabilities}."
        return f"The e-step does not calculate the probabilities and responsibilities correctly. {error_msg}"


class EStepTest(MethodTest):
    def define_tests(self):
        return [
            EStepShapeTest(),
            EStepOutputTest(),
        ]

    def define_method_name(self):
        return "e-step"


class MStepShapeTest(UnitTest):
    def __init__(self):
        self.num_clusters = 2
        self.max_iter = 10
        self.num_datapoints = 5
        self.data_dim = 3

    def test(self):
        data = torch.rand(self.num_datapoints, self.data_dim)
        kmeans = KMeans(self.num_clusters, self.max_iter)
        gmm = GaussianMixtureModels(
            self.num_clusters, kmeans, self.max_iter, full_init=False
        )

        self.centroids, self.covariances, self.mixing_coefficients = gmm.m_step(
            data,
            generate_rand_responsibilities(self.num_clusters, self.num_datapoints),
        )
        return (
            self.centroids.shape == (self.num_clusters, self.data_dim)
            and self.covariances.shape
            == (
                self.num_clusters,
                self.data_dim,
                self.data_dim,
            )
            and self.mixing_coefficients.shape == (self.num_clusters,)
        )

    def define_success_message(self):
        return f"Congratulations: the m-step produces the correct shape for the centroids, the covariances and the mixing coefficients."

    def define_failure_message(self):
        error_msg = ""
        if self.centroids.shape != (self.num_clusters, self.data_dim):
            error_msg += f"The centroids have the wrong shape. Expected {(self.num_clusters, self.data_dim)}, got {self.centroids.shape}."
        if self.covariances.shape != (self.num_clusters, self.data_dim, self.data_dim):
            error_msg += f"The covariances have the wrong shape. Expected {(self.num_clusters, self.data_dim, self.data_dim)}, got {self.covariances.shape}."
        if self.mixing_coefficients.shape != (self.num_clusters,):
            error_msg += f"The mixing coefficients have the wrong shape. Expected {(self.num_clusters,)}, got {self.mixing_coefficients.shape}."
        return f"The m-step does not produce the correct shape. {error_msg}"


class MStepOutputTest(UnitTest):
    def __init__(self):
        self.num_clusters = 2
        self.max_iter = 10
        self.num_datapoints = 5
        self.data_dim = 3
        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.data = torch.load("exercise_code/test/m_step_data.pt")
        self.responsibilities = torch.load(
            "exercise_code/test/m_step_responsibilities.pt"
        )
        self.centroids = torch.load("exercise_code/test/m_step_centroids.pt")
        self.covariances = torch.load("exercise_code/test/m_step_covariances.pt")
        self.mixing_coefficients = torch.load(
            "exercise_code/test/m_step_mixing_coefficients.pt"
        )

    def test(self):
        kmeans = KMeans(self.num_clusters, self.max_iter)
        gmm = GaussianMixtureModels(
            self.num_clusters, kmeans, self.max_iter, full_init=False
        )

        (
            self.out_centroids,
            self.out_covariances,
            self.out_mixing_coefficients,
        ) = gmm.m_step(
            self.data,
            self.responsibilities,
        )
        return (
            torch.all(torch.isclose(self.out_centroids, self.centroids))
            and torch.all(torch.isclose(self.out_covariances, self.covariances))
            and torch.all(
                torch.isclose(self.out_mixing_coefficients, self.mixing_coefficients)
            )
        )

    def define_success_message(self):
        return f"Congratulations: The m-step calculates the centroids, covariances, and mixing coefficients correctly."

    def define_failure_message(self):
        error_msg = ""
        if not torch.all(torch.isclose(self.out_centroids, self.centroids)):
            error_msg += f"The centroids were not calculated correctly. Expected {self.centroids}, got {self.out_centroids}."
        if not torch.all(torch.isclose(self.out_covariances, self.covariances)):
            error_msg += f"The covariances were not calculated correctly. Expected {self.covariances}, got {self.out_covariances}."
        if not torch.all(
            torch.isclose(self.out_mixing_coefficients, self.mixing_coefficients)
        ):
            error_msg += f"The mixing coefficients were not calculated correctly. Expected {self.mixing_coefficients}, got {self.out_mixing_coefficients}."
        return f"The m-step does not calculate the centroids, covariances, and mixing coefficients correctly. {error_msg}"


class MStepTest(MethodTest):
    def define_tests(self):
        return [
            MStepShapeTest(),
            MStepOutputTest(),
        ]

    def define_method_name(self):
        return "m-step"


class GaussianMixtureModelTest(ClassTest):
    def define_tests(self):
        return [
            EStepTest(),
            MStepTest(),
        ]

    def define_class_name(self):
        return "GaussianMixtureModel"


def test_gaussian_mixture_model():
    test = GaussianMixtureModelTest()
    return test_results_to_score(test())
