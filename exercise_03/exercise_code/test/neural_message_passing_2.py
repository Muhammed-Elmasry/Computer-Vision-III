from .base_tests import UnitTest, CompositeTest, test_results_to_score
from exercise_code.model.assign_net import AssignmentSimilarityNet
import torch

class Test_Shape_EdgeInitialization(UnitTest):
    def __init__(self):
        
        self.edge_in_dim = 6 # Adjust this in case your feature has a different dimension than edge_in_dim-1
        self.edge_dim = 9
        self.node_dim = 39
        self.assign_net = AssignmentSimilarityNet(reid_network=None, # Not needed since we work with precomputed features
                                     node_dim=self.node_dim, 
                                     edge_dim=self.edge_dim, 
                                     reid_dim=512, 
                                     edges_in_dim=self.edge_in_dim, 
                                     num_steps=10)
    def test(self):
        track_coords = torch.rand(11,4)
        current_coords = torch.rand(20,4)
        track_t = torch.rand(11)
        curr_t = torch.rand(20)
        num_tracks = track_coords.shape[0]
        num_boxes = current_coords.shape[0]
        edge = self.assign_net.compute_motion_edge_feats(track_coords, current_coords, track_t, curr_t)
        return edge.shape==(num_tracks, num_boxes, self.edge_in_dim-1)
        
    def define_success_message(self):
        return f"Congratulations: The shape of the Edge Initialization seems to be correct"
    def define_failure_message(self):
        return f"The shape of the Edge Initialization does not seem to be correct"


class NeuralMessagePassing(CompositeTest):
    def define_tests(self):
        return [
            Test_Shape_EdgeInitialization(),
        ]

    def define_method_name(self):
        return "neural_message_passing_2"

def test_neural_message_passing_2():
    test = NeuralMessagePassing()
    return test_results_to_score(test())