import torch
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from exercise_code.model.hungarian_tracker import Longterm_Hungarian_TrackerIoUReID

_UNMATCHED_COST = 255

class MPN_Tracker(Longterm_Hungarian_TrackerIoUReID):
    name = "MPN_Tracker"
    
    def __init__(self, assign_net, *args, **kwargs):
        self.assign_net = assign_net
        super().__init__(*args, **kwargs)
        

    def data_association(self, boxes, scores, pred_features):
        if self.tracks:
            device = next(self.assign_net.parameters()).device
            boxes = boxes.to(device)
            pred_features = pred_features.to(device)
            track_boxes = torch.stack([t.box for t in self.tracks], dim=0).to(device)
            track_features = torch.stack([t.get_feature() for t in self.tracks], dim=0).to(device)

            # Hacky way to recover the timestamps of boxes and tracks
            curr_t = self.im_index * torch.ones((pred_features.shape[0],)).to(device)
            track_t = torch.as_tensor([self.im_index - t.inactive - 1 for t in self.tracks]).to(device)

            ########################################################################
            # TODO: Forward pass through self.assign_net to obtain our costs.     #
            ########################################################################

            # Example forward pass (modify as per your network's requirements)
            # You might need to reshape or process the inputs as needed for your network
            pred_sim = self.assign_net(track_features, pred_features, track_boxes, boxes, track_t, curr_t)

            ########################################################################
            #                           END OF YOUR CODE                           #
            ########################################################################
            
            # Ensure pred_sim is properly formatted and processed
            pred_sim = pred_sim.detach().cpu().numpy()
            pred_sim = pred_sim[-1]  # Use predictions at last message passing step
            distance = 1 - pred_sim

            # Do not allow matches when sim < 0.5, to avoid low-confident associations
            distance = np.where(pred_sim < 0.5, _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes.cpu(), scores.cpu(), pred_features.cpu())

        else:
            # No tracks exist.
            self.add(boxes.cpu(), scores.cpu(), pred_features.cpu())
