
import torch
import numpy as np
seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

import time
import exercise_code.model.utils as utils
from exercise_code.model.evaluate_reid import evaluate_reid
from exercise_code.model.hard_mining import CombinedLoss, HardBatchMiningTripletLoss

MAX_EPOCH = 5
EPOCH_EVAL_FREQ = 56
PRINT_FREQ = 10

def train_classifier(model, optimizer, scheduler, train_loader, test_loader, device):
    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss() # <--- We start by using CrossEntropy loss.

    for epoch in range(MAX_EPOCH):
        losses = utils.MetricMeter()
        batch_time = utils.AverageMeter()
        end = time.time()
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Predict logits.
            imgs = data['img'].to(device)
            logits = model(imgs)
            # Get ground truth class distribution
            pids =  data['pid'].to(device)
            # Compute loss.
            loss = criterion(logits, pids)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            losses.update({'Loss': loss})
            if (batch_idx + 1) % PRINT_FREQ == 0:
                utils.print_statistics(
                    batch_idx,
                    num_batches,
                    epoch,
                    MAX_EPOCH,
                    batch_time,
                    losses)
            end = time.time()

        scheduler.step()    
        if (epoch + 1) % EPOCH_EVAL_FREQ == 0 or epoch == MAX_EPOCH - 1:
            rank1, mAP = evaluate_reid(model, test_loader)
            print(f'Epoch {epoch + 1}/{MAX_EPOCH}: Rank1: {rank1}, mAP: {mAP}')
    return model

def train_metric_mapping(model, optimizer, scheduler, train_loader, test_loader, device):

    num_batches = len(train_loader)
    criterion = CombinedLoss(0.3, 1.0, 1.0) # <--- Feel free to play around with the loss weights.

    for epoch in range(MAX_EPOCH):
        losses = utils.MetricMeter()
        batch_time = utils.AverageMeter()
        end = time.time()
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Predict output.
            imgs = data['img'].to(device)
            logits, features = model(imgs) # We have two outputs, logits are needed, as we enable the use of the CrossEntropy loss.
            # Get ground truth class distribution
            pids =  data['pid'].to(device)
            # Compute loss.
            loss, loss_summary = criterion(logits, features, pids)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            if (batch_idx + 1) % PRINT_FREQ == 0:
                utils.print_statistics(
                    batch_idx, 
                    num_batches, 
                    epoch, 
                    MAX_EPOCH, 
                    batch_time, 
                    losses)
            end = time.time()

        scheduler.step()    
        if (epoch + 1) % EPOCH_EVAL_FREQ == 0 or epoch == MAX_EPOCH - 1:
            rank1, mAP = evaluate_reid(model, test_loader)
            print('Epoch {0}/{1}: Rank1: {rank}, mAP: {map}'.format(
                        epoch + 1, MAX_EPOCH, rank=rank1, map=mAP))
    return model