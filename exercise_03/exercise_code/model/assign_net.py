import torch
import torch.nn as nn
from torch.nn import functional as F
from exercise_code.model.distance_metrics import cosine_distance

class BipartiteNeuralMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.0):
        super().__init__()

        edge_in_dim = 2 * node_dim + 2 * edge_dim  # since we always concatenate initial edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        node_in_dim = node_dim + edge_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        n_nodes_a, n_nodes_b, _ = edge_embeds.shape
        # Expand node embeddings to match the shape of the edge embeddings
        nodes_a_exp = nodes_a_embeds.unsqueeze(1).expand(-1, n_nodes_b, -1)
        nodes_b_exp = nodes_b_embeds.unsqueeze(0).expand(n_nodes_a, -1, -1)

        # Concatenate node and edge embeddings to form the input to the edge MLP
        self.edge_in = torch.cat([nodes_a_exp, nodes_b_exp, edge_embeds], dim=-1)
        
        # Pass the concatenated features through the edge MLP
        edge = self.edge_mlp(self.edge_in)
        return edge

    def node_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        # Sum edge features for all edges connected to each node
        sum_edges_a = edge_embeds.sum(dim=1)
        sum_edges_b = edge_embeds.sum(dim=0)

        # Concatenate the summed edge features with the node features
        self.nodes_a_in = torch.cat([nodes_a_embeds, sum_edges_a], dim=-1)
        self.nodes_b_in = torch.cat([nodes_b_embeds, sum_edges_b], dim=-1)
        
        # Pass the concatenated features through the node MLP
        nodes_a = self.node_mlp(self.nodes_a_in)
        nodes_b = self.node_mlp(self.nodes_b_in)
        return nodes_a, nodes_b

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        edge_embeds_latent = self.edge_update(edge_embeds, nodes_a_embeds, nodes_b_embeds)
        nodes_a_latent, nodes_b_latent = self.node_update(edge_embeds_latent, nodes_a_embeds, nodes_b_embeds)

        return edge_embeds_latent, nodes_a_latent, nodes_b_latent


class AssignmentSimilarityNet(nn.Module):
    def __init__(self, reid_network, node_dim, edge_dim, reid_dim, edges_in_dim, num_steps, dropout=0.0):
        super().__init__()
        self.reid_network = reid_network
        self.graph_net = BipartiteNeuralMessagePassingLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
        self.num_steps = num_steps
        self.cnn_linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(
            *[
                nn.Linear(edges_in_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        self.classifier = nn.Sequential(*[nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)])

    def compute_motion_edge_feats(self, track_coords, current_coords, track_t, curr_t):
        # Calculate the center points and sizes of the bounding boxes for tracks and current detections
        track_centers = (track_coords[:, :2] + track_coords[:, 2:]) / 2
        current_centers = (current_coords[:, :2] + current_coords[:, 2:]) / 2
        track_sizes = track_coords[:, 2:] - track_coords[:, :2]
        current_sizes = current_coords[:, 2:] - current_coords[:, :2]
        
        # Expand dimensions to allow for broadcasting
        tc_expanded = track_centers[:, None, :]  # Shape: (num_tracks, 1, 2)
        cc_expanded = current_centers[None, :, :]  # Shape: (1, num_boxes, 2)
        ts_expanded = track_sizes[:, None, :]  # Shape: (num_tracks, 1, 2)
        cs_expanded = current_sizes[None, :, :]  # Shape: (1, num_boxes, 2)
        
        # Calculate the differences in centers and sizes, and normalize by the sizes
        center_differences = 2 * (cc_expanded - tc_expanded) / (ts_expanded + cs_expanded)
        size_logs = torch.log((ts_expanded / cs_expanded).clamp(min=1e-6))
        
        # Calculate the time differences and expand to the right shape
        time_differences = (curr_t - track_t[:, None]).unsqueeze(-1)  # Shape: (num_tracks, num_boxes, 1)
        
        # Concatenate all features to create the edge feature tensor
        edge_feats = torch.cat([center_differences, size_logs, time_differences], dim=-1)
        
        # Ensure the final shape is as expected: (num_tracks, num_boxes, 5)
        assert edge_feats.shape == (track_coords.size(0), current_coords.size(0), 5)
        
        return edge_feats

    # ... (the rest of the class remains unchanged)


    def forward(self, track_app, current_app, track_coords, current_coords, track_t, curr_t):
        """
        Args:
            track_app: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_app: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)

        Returns:
            classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                             containing at entry (step, i, j) the unnormalized probability that track i and
                             detection j are a match, according to the classifier at the given neural message passing step
        """

        # Get initial edge embeddings
        edge_feats_app = cosine_distance(track_app, current_app)
        edge_feats_motion = self.compute_motion_edge_feats(track_coords, current_coords, track_t, curr_t)
        edge_feats = torch.cat((edge_feats_motion, edge_feats_app.unsqueeze(-1)), dim=-1)
        edge_embeds = self.edge_in_mlp(edge_feats)
        initial_edge_embeds = edge_embeds.clone()

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        node_embeds_track = F.relu(self.cnn_linear(track_app))
        node_embeds_curr = F.relu(self.cnn_linear(current_app))

        classified_edges = []
        for _ in range(self.num_steps):
            # Concat current edge embeds with initial edge embeds, increasing the feature dimension
            edge_embeds = torch.cat((edge_embeds, initial_edge_embeds), dim=-1)
            # Override edge_embeds, node_embeds
            edge_embeds, node_embeds_track, node_embeds_curr = self.graph_net(
                edge_embeds=edge_embeds,
                nodes_a_embeds=node_embeds_track,
                nodes_b_embeds=node_embeds_curr
            )
            # Run the classifier on edge embeddings
            classified_edges.append(self.classifier(edge_embeds))
        classified_edges = torch.stack(classified_edges).squeeze(-1)
        similarity = torch.sigmoid(classified_edges)
        return similarity
