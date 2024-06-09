import torch
from .base_tests import (
    MethodTest,
    test_results_to_score,
)
from exercise_code.test.hungarian_tracking_iou import HungarianTracking_MOTMETRICS

class HungarianTrackingTest(MethodTest):
    def define_tests(self):
        trackername = "Hungarian_TrackerIoUReID"
        return [
            HungarianTracking_MOTMETRICS('MOT16-02', trackername),
            # HungarianTracking_MOTMETRICS('MOT16-05', trackername),
        ]

    def define_method_name(self):
        return ""


def test_hungarian_tracking_ioureid():
    test = HungarianTrackingTest()
    return test_results_to_score(test())
