import tqdm
import torch
from torch.nn import functional as F

@torch.no_grad()
def compute_class_metric(P_pred, P, class_metrics = ('accuracy', 'recall', 'precision')):
    TP = (P & P_pred).sum().float()
    FP = (torch.logical_not(P) & P_pred).sum().float()
    TN = (torch.logical_not(P) & torch.logical_not(P_pred)).sum().float()
    FN = (P & torch.logical_not(P_pred)).sum().float()

    Acc = (TP + TN) / (TP + FP + TN + FN)
    Rec = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
    Prec = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict =  {'accuracy': Acc.item(), 'recall': Rec.item(), 'precision': Prec.item()}
    class_metrics_dict = {met_name: class_metrics_dict[met_name] for met_name in class_metrics}

    return class_metrics_dict

@torch.no_grad()
def update_metrics(metrics_accum, assign_sim, same_id, loss, batch_size):
    P_pred = (assign_sim[-1] > 0.5).view(-1)
    P = same_id[-1].view(-1)
    metrics = compute_class_metric(P_pred, P)

    for m_name, m_val in metrics.items():
        metrics_accum[m_name] += m_val / float(batch_size)
    metrics_accum['loss'] += loss.item()/ float(batch_size)

@torch.no_grad()
def print_metrics(metrics_accum, i, print_freq):
    if (i+1) %print_freq == 0 and i > 0:
        log_str = ". ".join([f"{m_name.capitalize()}: {m_val/ (print_freq if i !=0 else 1):.3f}" for m_name, m_val in metrics_accum.items()])
        print(f"Iter {i + 1}. " + log_str)
        

def train_assign_net_one_epoch(model, data_loader, optimizer, print_freq=200):    
    model.train()
    device = next(model.parameters()).device
    print(device)
    metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}
    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        optimizer.zero_grad()
        loss = 0

        # Since our model does not support automatic batching, we do manual
        # gradient accumulation        
        for sample in batch:
            past_frame, curr_frame = sample
            track_feats, track_coords, track_ids = past_frame['features'].to(device), past_frame['boxes'].to(device), past_frame['ids'].to(device)
            current_feats, current_coords, curr_ids = curr_frame['features'].to(device), curr_frame['boxes'].to(device), curr_frame['ids'].to(device)
            track_t, curr_t = past_frame['time'].to(device), curr_frame['time'].to(device)

            assign_sim=model.forward(track_app=track_feats, 
                                    current_app=current_feats, 
                                    track_coords=track_coords,
                                    current_coords=current_coords,
                                    track_t=track_t,
                                    curr_t=curr_t)

            # prepare ground truth signal for similarity, higher values (True) if tracks are similar
            same_id = (track_ids.view(-1, 1) == curr_ids.view(1, -1))
            same_id = same_id.unsqueeze(0).expand(assign_sim.shape[0], -1, -1)

            loss += F.binary_cross_entropy_with_logits(assign_sim, same_id.type(assign_sim.dtype), pos_weight=torch.as_tensor(20.))
            # Keep track of metrics
            update_metrics(metrics_accum, assign_sim, same_id, loss, len(batch))
        loss.backward()
        print_metrics(metrics_accum, i, print_freq)
        if (i+1) %print_freq == 0 and i > 0:
            metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}

        optimizer.step()
    model.eval()

@torch.no_grad()
def evaluate_assign_net(model, data_loader):
    loss = 0
    metrics_accum = {'loss': 0., 'accuracy': 0., 'recall': 0., 'precision': 0.}
    for i, batch in tqdm.tqdm(enumerate(data_loader)):

        device = next(model.parameters()).device
        # Since our model does not support automatic batching, we do manual
        # gradient accumulation        
        for sample in batch:
            past_frame, curr_frame = sample
            track_feats, track_coords, track_ids = past_frame['features'].to(device), past_frame['boxes'].to(device), past_frame['ids'].to(device)
            current_feats, current_coords, curr_ids = curr_frame['features'].to(device), curr_frame['boxes'].to(device), curr_frame['ids'].to(device)
            track_t, curr_t = past_frame['time'].to(device), curr_frame['time'].to(device)

            assign_sim=model.forward(track_app=track_feats, 
                                    current_app=current_feats, 
                                    track_coords=track_coords,
                                    current_coords=current_coords,
                                    track_t=track_t,
                                    curr_t=curr_t)

            # prepare ground truth signal for similarity, higher values (True) if tracks are similar
            same_id = (track_ids.view(-1, 1) == curr_ids.view(1, -1))
            same_id = same_id.unsqueeze(0).expand(assign_sim.shape[0], -1, -1)

            loss += F.binary_cross_entropy_with_logits(assign_sim, same_id.type(assign_sim.dtype), pos_weight=torch.as_tensor(20.))

            # Keep track of metrics
            update_metrics(metrics_accum, assign_sim, same_id, loss, len(batch)*len(data_loader))
    print_metrics(metrics_accum, i, 1)
    return metrics_accum
