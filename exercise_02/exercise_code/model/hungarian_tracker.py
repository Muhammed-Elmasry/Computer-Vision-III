import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from exercise_code.model.distance_metrics import cosine_distance
from exercise_code.model.tracker import TrackerIoU, TrackerIoUReID

_UNMATCHED_COST = 255.0

# Old Tracker
class Min_TrackerIoU(TrackerIoU):
    name = "Min_TrackerIoU"

    def data_association(self, boxes, scores):
        if self.tracks:
            distance = self.get_iou_distance(boxes)

            # update existing tracks
            remove_track_ids = []
            for t, dist in zip(self.tracks, distance):
                if np.isnan(dist).all():
                    remove_track_ids.append(t.id)
                else:
                    match_id = np.nanargmin(dist)
                    t.box = boxes[match_id]
            self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

            # add new tracks
            new_boxes = []
            new_scores = []
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
            self.add(new_boxes, new_scores)

        else:
            self.add(boxes, scores)


class Hungarian_TrackerIoU(TrackerIoU):
    name = "Hungarian_TrackerIoU"

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]

            distance = self.get_iou_distance(boxes)
            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)

            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.

            ########################################################################
            # TODO: Update existing tracks and remove unmatched tracks.           #
            #       Add new tracks if needed.                                     #
            ########################################################################

            # 1. Update existing tracks and remove unmatched tracks
            updated_tracks = []
            for i in range(len(row_idx)):
                if distance[row_idx[i], col_idx[i]] != _UNMATCHED_COST:
                    track_index = track_ids.index(self.tracks[row_idx[i]].id)
                    self.tracks[track_index].box = boxes[col_idx[i]]
                    self.tracks[track_index].score = scores[col_idx[i]]
                    updated_tracks.append(self.tracks[track_index])

            # 2. Remove unmatched tracks
            unmatched_tracks = [self.tracks[i] for i in range(len(self.tracks)) if i not in row_idx]
            for unmatched_track in unmatched_tracks:
                self.tracks.remove(unmatched_track)

            # 3. Add new tracks
            new_boxes = [boxes[i] for i in range(len(boxes)) if i not in col_idx]
            new_scores = [scores[i] for i in range(len(scores)) if i not in col_idx]
            self.add(new_boxes, new_scores)

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################

        else:
            # No tracks exist.
            self.add(boxes, scores)


class Hungarian_TrackerIoUReID(TrackerIoUReID):
    name = "Hungarian_TrackerIoUReID"

    def compute_distance_matrix(self, distance_app, distance_iou, alpha=0.0):
        UNMATCHED_COST = 255.0
        # Build cost matrix.
        assert np.alltrue(distance_app >= -0.1)
        assert np.alltrue(distance_app <= 1.1)

        combined_costs = alpha * distance_iou + (1 - alpha) * distance_app

        # Set all unmatched costs to _UNMATCHED_COST.
        distance = np.where(np.isnan(distance_iou), UNMATCHED_COST, combined_costs)
        return distance

    def data_association(self, boxes, scores, pred_features):

        if self.tracks:
            track_ids = [t.id for t in self.tracks]
        
            distance_iou = self.get_iou_distance(boxes)
            distance_app = self.get_app_distance(pred_features, metric_fn=cosine_distance) # This will use your similarity measure. Please use cosine_distance!
            distance = self.compute_distance_matrix(
                distance_app, distance_iou,
            )

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)

            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.

            ########################################################################
            # TODO:                                                                #
            # Update existing tracks and remove unmatched tracks.                  #
            # Reminder: If the costs are equal to _UNMATCHED_COST, it's NOT a      #
            # match. Be careful with overriding self.tracks, as past tracks will   #
            # be gone.                                                             #
            #                                                                      #
            # NOTE: Please update the feature of a track by using add_feature:     #
            # self.tracks[my_track_id].add_feature(pred_features[my_feat_index])   #
            # Reason: We use the mean feature from the last 10 frames for ReID.    #
            #                                                                      #
            # NOTE 1: self.tracks = ... <-- needs to be filled.                    #
            #                                                                      #
            # NOTE 2: # 1. costs == _UNMATCHED_COST -> remove.                     #
            # Optional: 2. tracks that have no match -> remove.                    #
            #                                                                      #
            # NOTE 3: Add new tracks. See TrackerIoU.                              #
            # new_boxes = []  # <-- needs to be filled.                            #
            # new_scores = []  # <-- needs to be filled.                           #
            ########################################################################


            pass

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################

            self.add(new_boxes, new_scores, new_features)
        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)


