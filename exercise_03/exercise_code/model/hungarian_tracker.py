import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from exercise_code.model.distance_metrics import cosine_distance
from exercise_code.model.tracker import TrackerIoUReID

_UNMATCHED_COST = 255.0


class Hungarian_TrackerIoUReID(TrackerIoUReID):
    name = "Hungarian_TrackerIoUReID"

    def compute_distance_matrix(self, distance_app, distance_iou, alpha=0.0):
        UNMATCHED_COST = 255.0
        # Build cost matrix.

        combined_costs = alpha * distance_iou + (1 - alpha) * distance_app

        # Set all unmatched costs to _UNMATCHED_COST.
        distance = np.where(np.isnan(distance_iou), UNMATCHED_COST, combined_costs)
        distance = np.where(distance_app > 0.1, UNMATCHED_COST, distance)

        return distance

    def data_association(self, boxes, scores, pred_features):

        if self.tracks:
            distance_iou = self.get_iou_distance(boxes)
            distance_app = self.get_app_distance(pred_features, metric_fn=cosine_distance) # This will use your similarity measure. Please use cosine_distance!
            distance = self.compute_distance_matrix(distance_app,
                                                    distance_iou)
            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)

        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):

        track_ids = [t.id for t in self.tracks]

        # update box position for matches and store matched ids
        matched_ids = []
        matched_box_ids = []
        for r_idx, c_idx in zip(row_idx, col_idx):
            if distance[r_idx, c_idx] != _UNMATCHED_COST:
                self.tracks[r_idx].box = boxes[c_idx]
                self.tracks[r_idx].add_feature(pred_features[c_idx])
                matched_ids.append(track_ids[r_idx])
                matched_box_ids.append(c_idx)

        # remove untracked tracks
        self.tracks = [t for t in self.tracks if t.id in matched_ids]

        # Add all new detections, that could not be matched
        new_boxes_idx = set(range(len(boxes))) - set(matched_box_ids)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)

class Longterm_Hungarian_TrackerIoUReID(Hungarian_TrackerIoUReID):
    name = "Longterm_Hungarian_TrackerIoUReID"

    def __init__(self, patience, *args, **kwargs):
        """ Add a patience parameter"""
        self.patience=patience
        super().__init__(*args, **kwargs)

    # -------------------- New code --------------------
    def update_results(self):
        """Only store boxes for tracks that are active"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            if t.inactive == 0: # Only change
                self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])
        self.im_index += 1 
    # ----------------------------------------------------------       
        
    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        track_ids = [t.id for t in self.tracks]

        # update box position for matches and store matched ids
        matched_ids = []
        matched_box_ids = []
        #r_idx, c_idx  = track_idx, box_idx
        for r_idx, c_idx in zip(row_idx, col_idx):
            if distance[r_idx, c_idx]  != _UNMATCHED_COST:
                self.tracks[r_idx].box = boxes[c_idx]
                self.tracks[r_idx].add_feature(pred_features[c_idx])
                matched_ids.append(track_ids[r_idx])
                matched_box_ids.append(c_idx)

        # -------------------- New line of code --------------------
        unmatched_track_ids = list(set(track_ids) - set(matched_ids))
        # ----------------------------------------------------------

        ########################################################################
        # TODO:                                                                #
        # 1: update the inactive attribute for all tracks                      #
        # 2: remove the tracks where inactive is bigger than self.patience     #
        ########################################################################
        
        # Update inactive attribute for all tracks
        for track_id in track_ids:
            if track_id in unmatched_track_ids:
                # If the track is unmatched, increase the inactive counter
                self.tracks[track_ids.index(track_id)].inactive += 1
            else:
                # If the track is matched, reset the inactive counter
                self.tracks[track_ids.index(track_id)].inactive = 0

        # Remove tracks where inactive is greater than self.patience
        self.tracks = [t for t in self.tracks if t.inactive <= self.patience]


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################      
        
        # Add all new detections, that could not be matched
        new_boxes_idx = set(range(len(boxes))) - set(matched_box_ids)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)