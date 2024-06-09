import torch
from .base_tests import (
    UnitTest,
)
from exercise_code.test.mot_metrics import evaluate_mot_accums, get_mot_accum
from exercise_code import MOT16Sequences
from pathlib import Path
import numpy as np        
import pandas as pd
import motmetrics as mm

root_dir = Path('./cv3dst/')
dataset_dir = root_dir.joinpath("datasets")
# ########################################################################
# # TODO:                                                                #
# # Nothing to do here                                                   #
# ########################################################################
from exercise_code import  Longterm_Hungarian_TrackerIoUReID, Hungarian_TrackerIoUReID, MPN_Tracker
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from exercise_code.test.run_tracker import run_tracker
class HungarianTracking_run_tracker_fast(UnitTest):
    def __init__(self, seq_name='MOT16-test', tracker_name="Hungarian_TrackerIoU", key="mota"):
        self.seq_name = seq_name
        MAX_PATIENCE = 20
        self.sequences = MOT16Sequences(self.seq_name, dataset_dir.joinpath("MOT16"), vis_threshold=0.)
        self.db = torch.load(dataset_dir.joinpath('reid_gnn', 'preprocessed_data_train_2.pth'))
        self.tracker_name = tracker_name
        if tracker_name == "Hungarian_TrackerIoUReID":
            self.tracker = Hungarian_TrackerIoUReID()
        elif tracker_name == "Longterm_Hungarian_TrackerIoUReID":
            self.tracker = Longterm_Hungarian_TrackerIoUReID(patience=MAX_PATIENCE)
        elif tracker_name == "MPN_Tracker":
            output_dir = root_dir.joinpath('exercise_03', 'models')
            assign_net = torch.load(output_dir.joinpath("assign_net.pth"))
            self.tracker = MPN_Tracker(assign_net=assign_net.eval(),patience=MAX_PATIENCE)
        self.key = key

    def test(self):
        results = run_tracker(self.sequences, self.db, self.tracker, device)
        ###################
        #### Takes too long on the CPU
        ###################
        # Evaluate the MOTMETRICS for a given dataset split provided by self.seq_name
        mot_accums = []
        for seq in self.sequences:
            if seq.no_gt:
                print(f"No GT evaluation data available.")
            else:
                mot_accums.append(get_mot_accum(results[str(seq)], seq))
        seq_sting = [str(s) for s in self.sequences if not s.no_gt]
        summary = evaluate_mot_accums(mot_accums, seq_sting)

        # Plot the values for debugging (second row is the references implementation)
        summary_render = pd.concat([summary.loc[seq_sting, [self.key]]], axis=0)
        str_summary = mm.io.render_summary(
            summary_render,
            namemap=mm.io.motchallenge_metric_names,
        )
        print(str_summary)

        # compare dataframes containing the metrics in columns and names in rows.
        self.eval_metrics = summary.loc[seq_sting, self.key].to_numpy()
        return int(100 * self.eval_metrics.mean())

    def define_message(self):
        return f"Your tracker {self.tracker_name} reached the mean {self.key} {self.eval_metrics.mean():.2f} on sequence {self.seq_name}.\nTest passed {self.eval_metrics.mean()*100:.0f}/100"

# ########################################################################
# #                           END OF YOUR CODE                           #
# ########################################################################

class HungarianTracking_MOTMETRICS(UnitTest):
    def __init__(self, seq_name, tracker_name):
        self.seq_name = seq_name
        self.tracker_name = tracker_name
        self.sequences = MOT16Sequences(self.seq_name, dataset_dir.joinpath("MOT16"))
        self.keys = ["mota", "idf1"]
        self.offsets = [-0.02, -0.04]
        self.summary_name = "exercise_code/test/"+self.tracker_name+"_summary.pth"
        self.results_name = "exercise_code/test/"+self.tracker_name+".pth"

    def test(self):
        # Load the tracking results (tracks of the sequences)
        results = torch.load(self.results_name)
        
        # Evaluate the MOTMETRICS for a given dataset split provided by self.seq_name
        mot_accums = []
        for seq in self.sequences:
            if seq.no_gt:
                print(f"No GT evaluation data available.")
            else:
                mot_accums.append(get_mot_accum(results[str(seq)], seq))
        seq_sting = [str(s) for s in self.sequences if not s.no_gt]
        summary = evaluate_mot_accums(mot_accums, seq_sting)
        #torch.save(summary,self.summary_name)
        # Load the evaluated MOTMETRICS of reference implementation for comparison
        summary_stored = torch.load(self.summary_name)
        summary_stored = summary_stored.loc[seq_sting,:]

        # Plot the values for debugging (second row is the references implementation)
        summary_render = pd.concat([summary, summary_stored], axis=0)
        str_summary = mm.io.render_summary(
            summary_render,
            namemap=mm.io.motchallenge_metric_names,
        )
        print(str_summary)

        # compare dataframes containing the metrics in columns and names in rows.
        self.eval_metrics = summary.loc[seq_sting, self.keys].to_numpy()
        self.eval_metrics_stored = summary_stored.loc[seq_sting, self.keys].to_numpy() + np.array([self.offsets])
        return np.all(np.greater_equal(self.eval_metrics, self.eval_metrics_stored))
         
    def define_exception_message(self):
        return f"Exception"

    def define_success_message(self):
        return f"Congratulations: {self.tracker_name} seems to be correct for sequence {self.seq_name} based on the metrics: {' '.join(self.keys)}."

    def define_failure_message(self):
        return f"{self.tracker_name} does not seem to be correct for {self.seq_name} based on the metrics: {' '.join(self.keys)}."

class HungarianTracking_MOTMETRICS_soft(UnitTest):
    def __init__(self, seq_name='MOT16-reid', tracker_name="Hungarian_TrackerIoU", key="mota"):
        self.seq_name = seq_name
        self.tracker_name = tracker_name
        self.sequences = MOT16Sequences(self.seq_name, dataset_dir.joinpath("MOT16"))
        self.key = key
        self.results_name = "exercise_code/test/"+self.tracker_name+".pth"

    def test(self):
        # Load the tracking results (tracks of the sequences)
        results = torch.load(self.results_name)
        
        # Evaluate the MOTMETRICS for a given dataset split provided by self.seq_name
        mot_accums = []
        for seq in self.sequences:
            if seq.no_gt:
                print(f"No GT evaluation data available.")
            else:
                mot_accums.append(get_mot_accum(results[str(seq)], seq))
        seq_sting = [str(s) for s in self.sequences if not s.no_gt]
        summary = evaluate_mot_accums(mot_accums, seq_sting)

        # Plot the values for debugging (second row is the references implementation)
        summary_render = pd.concat([summary.loc[seq_sting, [self.key]]], axis=0)
        str_summary = mm.io.render_summary(
            summary_render,
            namemap=mm.io.motchallenge_metric_names,
        )
        print(str_summary)

        # compare dataframes containing the metrics in columns and names in rows.
        self.eval_metrics = summary.loc[seq_sting, self.key].to_numpy()
        return int(100 * self.eval_metrics.mean())
        
    def define_message(self):
        return f"Your tracker {self.tracker_name} reached the mean {self.key} {self.eval_metrics.mean():.2f} on sequence {self.seq_name}.\nTest passed {self.eval_metrics.mean()*100:.0f}/100"
