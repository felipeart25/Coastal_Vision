import torch
import torch.nn as nn
import torchmetrics

import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, apply_sigmoid=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets):
        if self.apply_sigmoid:
            inputs = torch.sigmoid(inputs)

        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice
    
def dice_score(y_pred, y_true, delta=0.5, eps=1e-6):
    """
    Computes the Sørensen–Dice coefficient.
    
    Args:
        y_pred (Tensor): predicted probabilities (B, 1, H, W)
        y_true (Tensor): ground truth binary mask (B, 1, H, W)
        delta (float): weight on FN vs FP
        eps (float): small value to avoid division by 0
        
    Returns:
        dice (Tensor): Dice score for each sample in batch (B,)
    """
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    y_pred = (y_pred > 0.5).float()  # binarize for evaluation

    y_pred = y_pred.view(y_pred.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)

    tp = (y_true * y_pred).sum(dim=1)
    fn = (y_true * (1 - y_pred)).sum(dim=1)
    fp = ((1 - y_true) * y_pred).sum(dim=1)

    dice = (tp + eps) / (tp + delta * fn + (1 - delta) * fp + eps)
    return dice
    
def compute_iou(pred, target):
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = ((pred_bin + target_bin) > 0).float().sum(dim=(1, 2, 3)) + 1e-6

    return (intersection / union).mean()

def compute_precision(pred, target):
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fp = ((pred_bin == 1) & (target_bin == 0)).float().sum(dim=(1, 2, 3)) + 1e-6

    return (tp / (tp + fp)).mean()

def compute_recall(pred, target):
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    fn = ((pred_bin == 0) & (target_bin == 1)).float().sum(dim=(1, 2, 3)) + 1e-6

    return (tp / (tp + fn)).mean()



class UnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.75, eps=1e-7):
        """
        Unified Focal Loss for binary segmentation (single channel).

        Args:
            weight (float): lambda parameter balancing the two terms.
            delta (float): balance between FP and FN in Tversky.
            gamma (float): focal parameter.
            eps (float): small value to avoid division by zero.
        """
        super(UnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1. - self.eps)

        # Flatten
        y_true_f = y_true.view(y_true.size(0), -1)
        y_pred_f = y_pred.view(y_pred.size(0), -1)

        # True Positives, False Positives & False Negatives
        tp = (y_true_f * y_pred_f).sum(dim=1)
        fn = (y_true_f * (1 - y_pred_f)).sum(dim=1)
        fp = ((1 - y_true_f) * y_pred_f).sum(dim=1)

        # Focal Tversky Loss
        tversky = (tp + self.eps) / (tp + self.delta * fn + (1 - self.delta) * fp + self.eps)
        focal_tversky = (1 - tversky) ** self.gamma

        # Focal BCE Loss
        bce = - (y_true_f * torch.log(y_pred_f) + (1 - y_true_f) * torch.log(1 - y_pred_f))
        focal_bce = (1 - y_pred_f) ** self.gamma * bce
        focal_bce = focal_bce.mean(dim=1)  # per sample

        # Combine
        loss = self.weight * focal_tversky + (1 - self.weight) * focal_bce
        return loss.mean()

class UnifiedFocalLossSample(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.75, eps=1e-7):
        """
        Unified Focal Loss for binary segmentation (single channel).

        Args:
            weight (float): lambda parameter balancing the two terms.
            delta (float): balance between FP and FN in Tversky.
            gamma (float): focal parameter.
            eps (float): small value to avoid division by zero.
        """
        super(UnifiedFocalLossSample, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1. - self.eps)

        # Flatten
        y_true_f = y_true.view(y_true.size(0), -1)
        y_pred_f = y_pred.view(y_pred.size(0), -1)

        # True Positives, False Positives & False Negatives
        tp = (y_true_f * y_pred_f).sum(dim=1)
        fn = (y_true_f * (1 - y_pred_f)).sum(dim=1)
        fp = ((1 - y_true_f) * y_pred_f).sum(dim=1)

        # Focal Tversky Loss
        tversky = (tp + self.eps) / (tp + self.delta * fn + (1 - self.delta) * fp + self.eps)
        focal_tversky = (1 - tversky) ** self.gamma

        # Focal BCE Loss
        bce = - (y_true_f * torch.log(y_pred_f) + (1 - y_true_f) * torch.log(1 - y_pred_f))
        focal_bce = (1 - y_pred_f) ** self.gamma * bce
        focal_bce = focal_bce.mean(dim=1)  # per sample

        # Combine
        loss = self.weight * focal_tversky + (1 - self.weight) * focal_bce
        return loss
    
def dice_score_land_only(y_pred, y_true, delta=0.5, eps=1e-6):
    """
    Computes Dice score over land (foreground) pixels for each sample in batch.

    Args:
        y_pred: predicted probabilities (B, 1, H, W)
        y_true: ground truth binary masks (B, 1, H, W)

    Returns:
        Tensor of shape (B,) with Dice per sample (only over foreground).
    """
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    y_pred_bin = (y_pred > 0.5).float()

    B = y_pred.size(0)
    scores = []

    for i in range(B):
        y_p = y_pred_bin[i].view(-1)
        y_t = y_true[i].view(-1)

        mask = y_t == 1
        if mask.sum() == 0:
            scores.append(torch.tensor(1.0, device=y_pred.device))  # or 0.0 if you prefer
            continue

        y_p_m = y_p[mask]
        y_t_m = y_t[mask]

        tp = (y_t_m * y_p_m).sum()
        fn = (y_t_m * (1 - y_p_m)).sum()
        fp = ((1 - y_t_m) * y_p_m).sum()

        dice = (tp + eps) / (tp + delta * fn + (1 - delta) * fp + eps)
        scores.append(dice)

    return torch.stack(scores)  # shape: (B,)

from scipy.ndimage import binary_dilation
import numpy as np

def buffer_mask(y_true_batch, buffer_size=10):
    """
    Generates shoreline buffer masks for each frame in a batch of sequences.

    Args:
        y_true_batch (Tensor): (B, T, 1, H, W) ground truth binary masks
        buffer_size (int): pixel width of the buffer zone

    Returns:
        buffer_masks (Tensor): (B, T, H, W) boolean masks indicating buffer zones
    """
  # Add a dimension for future sequence

    T, C, H, W = y_true_batch.shape
    assert C == 1, "Expected input to have 1 channel"
    
    # Convert to numpy for processing
    y_true_np = y_true_batch.squeeze(1).cpu().numpy()  # Shape: (B, T, H, W)

    buffer_masks = np.zeros((T, H, W), dtype=bool)


    for t in range(T):      # Loop over time
        gt = y_true_np[t].astype(np.float32)

        # Compute edges using gradient
        grad_y, grad_x = np.gradient(gt)
        edge_map = (np.abs(grad_y) + np.abs(grad_x)) > 0

        # Create buffer via dilation
        buffer = binary_dilation(edge_map, iterations=buffer_size)

        buffer_masks[t] = buffer

    return torch.tensor(buffer_masks, dtype=torch.bool, device=y_true_batch.device)

def buffer_mask_vec(y_true_batch, buffer_size=10):
    """
    Generates shoreline buffer masks for a batch of sequences.

    Args:
        y_true_batch (Tensor): (B, T, 1, H, W) ground truth binary masks
        buffer_size (int): pixel width of the buffer zone

    Returns:
        buffer_masks (Tensor): (B, T, 1, H, W) float masks indicating buffer zones (1=buffer, 0=background)
    """
    if y_true_batch.dim() == 4:
        y_true_batch = y_true_batch.unsqueeze(1)

    B, T, C, H, W = y_true_batch.shape
    assert C == 1, "Expected input to have 1 channel"

    # Convert to numpy for processing
    y_true_np = y_true_batch.squeeze(2).cpu().numpy()  # Shape: (B, T, H, W)

    buffer_masks = np.zeros((B, T, H, W), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            gt = y_true_np[b, t].astype(np.float32)

            # Compute edges using gradient
            grad_y, grad_x = np.gradient(gt)
            edge_map = (np.abs(grad_y) + np.abs(grad_x)) > 0

            # Create buffer via dilation
            buffer = binary_dilation(edge_map, iterations=buffer_size)

            buffer_masks[b, t] = buffer.astype(np.float32)

    # Convert back to torch tensor, add channel dim (B, T, 1, H, W)
    return torch.tensor(buffer_masks, dtype=torch.float32, device=y_true_batch.device).unsqueeze(2)

def dice_score_shoreline_buffer(y_pred, y_true, buffer_mask, delta=0.5, eps=1e-6):
    """
    Computes Dice score in buffer region around shoreline.

    Args:
        y_pred: (B, 1, H, W)
        y_true: (B, 1, H, W)
    """
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    y_pred_bin = (y_pred > 0.5).float()
    
    # Check

    #flatten

    T, C, H, W = y_pred.shape
    y_pred_flat = y_pred_bin.view(T, C, H, W)
    y_true_flat = y_true.view(T, C, H, W)
    buffer_mask_flat = buffer_mask.view(T, C, H, W)

    B = y_true.size(0)
    scores = []

    for i in range(B):

        pred_i = y_pred_flat[i, 0].squeeze()
        

        true_i = y_true_flat[i, 0].squeeze()
        mask = buffer_mask_flat[i, 0]

        gt_buffer = true_i[mask]
        pred_buffer = pred_i[mask]

        tp = (gt_buffer * pred_buffer).sum()
        fn = (gt_buffer * (1 - pred_buffer)).sum()
        fp = ((1 - gt_buffer) * pred_buffer).sum()

        dice = (tp + eps) / (tp + delta * fn + (1 - delta) * fp + eps)
        scores.append(dice)

    return torch.tensor(scores, device=y_pred.device)

def buffer_zone_metrics(y_pred, y_true, buffer_mask, eps=1e-6):
    """
    Computes IoU, Recall, and Precision inside the buffer zone.

    Args:
        y_pred: (B, 1, H, W) predicted probabilities
        y_true: (B, 1, H, W) ground truth binary masks
        buffer_masks: (B, H, W) boolean masks indicating buffer zones

    Returns:
        ious, recalls, precisions: Tensors of shape (B,) with per-sample scores
    """
    y_pred_bin = (torch.clamp(y_pred, eps, 1 - eps) > 0.5).float()

    T, C, H, W = y_pred.shape
    y_pred_flat = y_pred_bin.view(T, C, H, W)
    y_true_flat = y_true.view(T, C, H, W)
    buffer_mask_flat = buffer_mask.view(T, C, H, W)

    B = y_true.size(0)
    ious, recalls, precisions = [], [], []

    for i in range(B):
        pred_i = y_pred_flat[i, 0].squeeze()
        true_i = y_true_flat[i, 0].squeeze()
        mask = buffer_mask_flat[i, 0]

        gt_buffer = true_i[mask]
        pred_buffer = pred_i[mask]

        tp = (gt_buffer * pred_buffer).sum()
        fn = (gt_buffer * (1 - pred_buffer)).sum()
        fp = ((1 - gt_buffer) * pred_buffer).sum()
        union = tp + fp + fn

        # Metrics
        iou = (tp + eps) / (union + eps)
        recall = (tp + eps) / (tp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)

        ious.append(iou)
        recalls.append(recall)
        precisions.append(precision)

    return (
        torch.tensor(ious, device=y_pred.device),
        torch.tensor(recalls, device=y_pred.device),
        torch.tensor(precisions, device=y_pred.device)
    )






