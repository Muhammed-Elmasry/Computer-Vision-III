
from .base_tests import (
    MethodTest,
    test_results_to_score,
)
from exercise_code.test.hungarian_tracking import HungarianTracking_MOTMETRICS, HungarianTracking_MOTMETRICS_soft

class HungarianTrackingTest(MethodTest):
    def define_tests(self, trackername):
        return [
            HungarianTracking_MOTMETRICS('MOT16-02', trackername),
            HungarianTracking_MOTMETRICS('MOT16-05', trackername),
            HungarianTracking_MOTMETRICS('MOT16-09', trackername),
            HungarianTracking_MOTMETRICS('MOT16-11', trackername),
        ]

    def define_method_name(self):
        return ""


def test_longterm_hungarian_tracking_ioureid():
    tracker_name = "Longterm_Hungarian_TrackerIoUReID"

    #1
    test = HungarianTrackingTest(tracker_name)
    score = test_results_to_score(test(), verbose=False)

    #2
    test_soft = HungarianTracking_MOTMETRICS_soft(tracker_name=tracker_name, key="mota")
    score_soft = test_soft.test()
    print(test_soft.define_message())

    score_total = (score + score_soft)/2
    print("Score: %d/100" % score_total)
    return score_total
