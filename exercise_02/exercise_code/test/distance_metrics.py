import torch
from .base_tests import UnitTest, CompositeTest, test_results_to_score

from exercise_code.model.distance_metrics import euclidean_squared_distance, cosine_distance


########################################################################
# TODO:                                                                #
# Nothing to do here                                                   #
########################################################################

pass

########################################################################
#                           END OF YOUR CODE                           #
########################################################################


class EuclideanDistanceTest(UnitTest):
    def __init__(self):
        self.feature_1 = torch.load("exercise_code/test/feature_1.pth")
        self.feature_2 = torch.load("exercise_code/test/feature_2.pth")

        self.euclidian_distance = torch.load("exercise_code/test/euclidean_distance.pth")

    def test(self):
        return torch.all(
            torch.isclose(
                self.euclidian_distance,
                euclidean_squared_distance(
                    self.feature_1, self.feature_2
                ),
                1.0e-5,
            )
        )

    def define_success_message(self):
        return f"Congratulations: The euclidean distance was computed correctly."

    def define_failure_message(self):
        return f"The euclidean distance was not computed correctly"


class CosineDistanceTest(UnitTest):
    def __init__(self):
        self.feature_1 = torch.load("exercise_code/test/feature_1.pth")
        self.feature_2 = torch.load("exercise_code/test/feature_2.pth")

        self.cos_distance = torch.load("exercise_code/test/cosine_distance.pth")

    def test(self):
        return torch.all(
            torch.isclose(
                self.cos_distance,
                cosine_distance(
                    self.feature_1, self.feature_2
                ),
                1.0e-5,
            )
        )

    def define_success_message(self):
        return f"Congratulations: The cosine distance was computed correctly."

    def define_failure_message(self):
        return f"The cosine distance was not computed correctly"


class DistanceMetricsTest(CompositeTest):
    def define_tests(self):
        return [
            EuclideanDistanceTest(),
            CosineDistanceTest(),
        ]


def test_distance_metric():
    test = DistanceMetricsTest()
    return test_results_to_score(test())
