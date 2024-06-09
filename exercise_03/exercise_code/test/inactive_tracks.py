import torch
from .base_tests import UnitTest, CompositeTest,test_results_to_score
from exercise_code import Longterm_Hungarian_TrackerIoUReID
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from exercise_code.model.hungarian_tracker import _UNMATCHED_COST

class Test_Inactive_Remove(UnitTest):
    def __init__(self):
        self.patience = 5
        self.tracker = Longterm_Hungarian_TrackerIoUReID(patience=self.patience)
        self.boxes = torch.load("exercise_code/test/boxes.pth")
        self.scores = torch.load("exercise_code/test/scores.pth")
        self.pred_features = torch.load("exercise_code/test/pred_features.pth")
        self.tracker.data_association(self.boxes, self.scores, self.pred_features)
        self.len_tracks = len(self.tracker.tracks)
    
    def test(self):
        self.boxes = self.boxes[:1]
        self.scores = self.scores[:1]
        self.pred_features = self.pred_features[:1]
        for i in range(self.patience+1):
            distance = torch.ones(len(self.tracker.tracks), 1)*_UNMATCHED_COST
            row_idx = range(len(self.tracker.tracks))
            col_idx = range(1)
            self.tracker.update_tracks(row_idx, col_idx, distance, self.boxes, self.scores, self.pred_features)
        
        return len(self.tracker.tracks)==self.patience+1
        
    def define_success_message(self):
        return f"Congratulations: Tracks have been removed after patience was reached."
    def define_failure_message(self):
        return f"Removing failed after patience was reached."

class Test_Inactive_Counter(UnitTest):
    def __init__(self):
        self.patience = 5
        self.tracker = Longterm_Hungarian_TrackerIoUReID(patience=self.patience)
        self.boxes = torch.load("exercise_code/test/boxes.pth")
        self.scores = torch.load("exercise_code/test/scores.pth")
        self.pred_features = torch.load("exercise_code/test/pred_features.pth")
        self.tracker.data_association(self.boxes, self.scores, self.pred_features)
        self.len_tracks = len(self.tracker.tracks)
    
    def test(self):
        self.boxes = self.boxes[:1]
        self.scores = self.scores[:1]
        self.pred_features = self.pred_features[:1]
        for i in range(self.patience):
            distance = torch.ones(len(self.tracker.tracks), 1)*_UNMATCHED_COST
            row_idx = range(len(self.tracker.tracks))
            col_idx = range(1)
            self.tracker.update_tracks(row_idx, col_idx, distance, self.boxes, self.scores, self.pred_features)
        
        inactive_flags = [self.tracker.tracks[i].inactive for i in range(self.len_tracks)]
        return np.allclose(inactive_flags, self.patience)
        
    def define_success_message(self):
        return f"Congratulations: No tracks have been removed before patience was reached. Counter is correct."
    def define_failure_message(self):
        return f"Tracks have been removed before patience was reached. Note: if counter==patience, tracks are kept."

class Test_Keep_Tracks(UnitTest):
    def __init__(self):
        self.tracker = Longterm_Hungarian_TrackerIoUReID(patience=10)
        self.boxes = torch.load("exercise_code/test/boxes.pth")
        self.scores = torch.load("exercise_code/test/scores.pth")
        self.pred_features = torch.load("exercise_code/test/pred_features.pth")
        self.tracker.data_association(self.boxes, self.scores, self.pred_features)
        self.len_tracks = len(self.tracker.tracks)
    
    def test(self):
        self.boxes = self.boxes[:self.len_tracks-2]
        self.scores = self.scores[:self.len_tracks-2]
        self.pred_features = self.pred_features[:self.len_tracks-2]
        distance = torch.rand(self.len_tracks, len(self.boxes))
        row_idx, col_idx = linear_assignment(distance)
        self.tracker.update_tracks(row_idx, col_idx, distance, self.boxes, self.scores, self.pred_features)
        return self.len_tracks==len(self.tracker.tracks)
        
    def define_success_message(self):
        return f"Congratulations: No tracks have been removed before patience was reached."
    def define_failure_message(self):
        return f"Tracks have been removed before patience was reached."

class InactiveTracksTest(CompositeTest):
    def define_tests(self):
        return [
            Test_Keep_Tracks(),
            Test_Inactive_Counter(),
            Test_Inactive_Remove()
        ]

    def define_method_name(self):
        return "inactive_tracks"

def test_inactive_tracks():
    test = InactiveTracksTest()
    return test_results_to_score(test())