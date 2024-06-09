import torch
from exercise_code.model.distance_metrics import euclidean_squared_distance



class HardBatchMiningTripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining of samples in a batch.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def compute_distance_pairs(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        Output:
            distance_positive_pairs (torch.Tensor): distance (to one pos. sample per sample) with shape (batch_size).
            distance_negative_pairs (torch.Tensor): distance (to one neg. sample per sample) with shape (batch_size).
        """
        batch_size = inputs.size(0)

        # Compute the pairwise Euclidean distance between all feature vectors.
        distance_matrix = torch.cdist(inputs, inputs, p=2)  # Euclidean distance
        distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()

        # For each sample, find the hardest positive and hardest negative samples in the batch.
        distance_positive_pairs, distance_negative_pairs = [], []
        for i in range(batch_size):
            # Positive pairs should be as close as possible.
            positive_mask = (targets == targets[i]) & (targets != -1)  # Positive samples have the same class.
            negative_mask = (targets != targets[i]) & (targets != -1)  # Negative samples have a different class.

            if torch.sum(positive_mask) > 1:
                distance_positive_pairs.append(torch.min(distance_matrix[i, positive_mask]))
            else:
                # If there's only one positive sample (itself), use the maximum distance.
                distance_positive_pairs.append(torch.max(distance_matrix[i, positive_mask]))

            if torch.sum(negative_mask) > 0:
                # Negative pairs should be quite far apart.
                distance_negative_pairs.append(torch.max(distance_matrix[i, negative_mask]))
            else:
                # If there are no negative samples, use the minimum distance.
                distance_negative_pairs.append(torch.min(distance_matrix[i, ~negative_mask]))

        # Convert the created lists into 1D PyTorch tensors.
        distance_positive_pairs = torch.stack(distance_positive_pairs)
        distance_negative_pairs = torch.stack(distance_negative_pairs)

        return distance_positive_pairs, distance_negative_pairs

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        Output:
            loss (torch.Tensor): scalar loss, reduction by mean along the batch_size.
        """

        distance_positive_pairs, distance_negative_pairs = self.compute_distance_pairs(inputs, targets)
        distance_positive_pairs = distance_positive_pairs.to(inputs.device)
        distance_negative_pairs = distance_negative_pairs.to(inputs.device)

        # The ranking loss will compute the triplet loss with the margin.
        # loss = max(0, -1*(neg_dist - pos_dist) + margin)
        y = torch.ones_like(distance_negative_pairs)
        # one in y indicates that the first input should be ranked higher than the second input, which is true for all the samples
        loss = self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)

        return loss

class CombinedLoss(object):
  def __init__(self, margin=0.3, weight_triplet=1.0, weight_ce=1.0):
      super(CombinedLoss, self).__init__()
      self.triplet_loss = HardBatchMiningTripletLoss() # <--- Your code is used here!
      self.cross_entropy = torch.nn.CrossEntropyLoss()
      self.weight_triplet = weight_triplet
      self.weight_ce = weight_ce

  def __call__(self, logits, features, gt_pids):
      loss = 0.0
      loss_summary = {}
      if self.weight_triplet > 0.0:
        loss_t = self.triplet_loss(features, gt_pids) * self.weight_triplet
        loss += loss_t
        loss_summary['Triplet Loss'] = loss_t
      
      if self.weight_ce > 0.0:
        loss_ce = self.cross_entropy(logits, gt_pids) * self.weight_ce
        loss += loss_ce
        loss_summary['CE Loss'] = loss_ce

      loss_summary['Loss'] = loss
      return loss, loss_summary