from .base_tests import UnitTest, CompositeTest, test_results_to_score
from exercise_code.model.mpn_tracker import MPN_Tracker
from exercise_code.model.assign_net import AssignmentSimilarityNet
import torch

class Test_ForwardPass(UnitTest):
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
        MAX_PATIENCE = 20
        self.tracker = MPN_Tracker(assign_net=self.assign_net.eval(), obj_detect=None, patience=MAX_PATIENCE)
        self.boxes = torch.load("exercise_code/test/boxes.pth")
        self.scores = torch.load("exercise_code/test/scores.pth")
        self.pred_features = torch.load("exercise_code/test/pred_features.pth")

    def test(self):
        self.tracker.add(self.boxes, self.scores, self.pred_features)
        self.tracker.data_association(self.boxes, self.scores, self.pred_features)
        return True
        
    def define_success_message(self):
        return f"Congratulations: The forward pass seems to work"
    def define_failure_message(self):
        return f"The forward pass is not working"


class ForwardPassTest(CompositeTest):
    def define_tests(self):
        return [
            Test_ForwardPass(),
        ]

    def define_method_name(self):
        return "mpn_tracking_forward_pass"

def test_mpn_tracking_forward_pass():
    test = ForwardPassTest()
    return test_results_to_score(test())