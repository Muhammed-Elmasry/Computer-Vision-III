import torch
from .base_tests import UnitTest, MethodTest, test_results_to_score

from exercise_code.model.hard_mining import HardBatchMiningTripletLoss

########################################################################
# TODO:                                                                #
# Nothing to do here                                                   #
########################################################################

pass

########################################################################
#                           END OF YOUR CODE                           #
########################################################################

class HardMiningShapeTest(UnitTest):
    def __init__(self):
        self.batch_size = 20
        self.feat_dim = 5

        self.feature_matrix = torch.rand(self.batch_size, self.feat_dim)
        self.labels = torch.randint(0, 2, (self.batch_size,))

        self.hard_mininig_triplet_loss = HardBatchMiningTripletLoss()

    def test(self):
        distance_positive_pairs, distance_negative_pairs = self.hard_mininig_triplet_loss.compute_distance_pairs(self.feature_matrix, self.labels)

        return (distance_positive_pairs.shape == torch.Size((self.batch_size,))) and (distance_negative_pairs.shape == torch.Size((self.batch_size,)))

    def define_success_message(self):
        return f"Congratulations: HardBatchMining returns the correct shape."

    def define_failure_message(self):
        return f"HardBatchMining does not return the correct shape."

class HardMiningOutputTest(UnitTest):
    def __init__(self):
        self.feature_matrix = torch.load("exercise_code/test/feature_matrix.pth")
        self.labels = torch.load("exercise_code/test/labels.pth")

        self.distance_positive_pairs = torch.load("exercise_code/test/distance_positive_pairs.pth")
        self.distance_negative_pairs = torch.load("exercise_code/test/distance_negative_pairs.pth")
        
        self.hard_mininig_triplet_loss = HardBatchMiningTripletLoss()

    def test(self):
        distance_positive_pairs, distance_negative_pairs = self.hard_mininig_triplet_loss.compute_distance_pairs(self.feature_matrix, self.labels)
        return torch.all(
            torch.isclose(
                self.distance_positive_pairs,
                distance_positive_pairs,
                1.0e-5,
            )
        ) and torch.all(
            torch.isclose(
                self.distance_negative_pairs,
                distance_negative_pairs,
                1.0e-5,
            )
        )

    def define_success_message(self):
        return f"Congratulations: HardBatchMining seems to be correct."

    def define_failure_message(self):
        return f"HardBatchMining does not seem to be correct."


class HardMiningTest(MethodTest):
    def define_tests(self):
        return [
            HardMiningShapeTest(),
            HardMiningOutputTest(),
        ]

    def define_method_name(self):
        return "compute_distance_pairs"


def test_hard_mining():
    test = HardMiningTest()
    return test_results_to_score(test())
