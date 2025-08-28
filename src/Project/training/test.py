import torch
import numpy as np
import wandb
from utils.metrics import dice_score, dice_score_shoreline_buffer, buffer_zone_metrics, buffer_mask
from utils.Analysis import diagnose, buffer_zone_vis

def test_metrics(model, test_loader, device, test_data, future_seq=1):
    model.eval()

    #### Define variables to store #####
    all_inputs = []
    all_targets = []
    all_outputs = []
    all_wave_contexts = []
    all_wave_contexts_future = []
    all_buffer_masks = []

    dice_scores = []
    dice_buffer_s = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    idx = []

    # Initialize per-frame metric lists
    per_frame_dice = [[] for _ in range(future_seq)]
    per_frame_dice_buffer = [[] for _ in range(future_seq)]
    per_frame_iou = [[] for _ in range(future_seq)]
    per_frame_precision = [[] for _ in range(future_seq)]
    per_frame_recall = [[] for _ in range(future_seq)]

    #### Initialize metrics ####
    total_samples = 0
    dice_s = 0
    dice_b = 0
    iou_s = 0
    precision_s = 0
    recall_s = 0

    #### Disable gradient calculation ####
    with torch.no_grad():

        #### Loop through test data per batch ####
        for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(test_loader):

            #### Move data to device ####
            slow_input = slow_input.to(device)
            fast_input = fast_input.to(device)
            target_seq = target_seq.to(device)
            fast_fut_input = fast_fut_input.to(device)

            #### Forward pass ####
            output = model(slow_input, fast_input, fast_fut_input, future_seq=future_seq)

            # Handle targets based on strategy
            if future_seq == 1:
                target_seq = target_seq[:, :future_seq]
            else:
                target_seq = target_seq[:, :future_seq]      

            B = output.shape[0]
            total_samples += B

            # Dice over full sequence
            dice = dice_score(output, target_seq)
            dice_s += dice.sum().item()

            

            # Cut off future wave data
            cutoff = future_seq * (fast_fut_input.size(1) // target_seq.size(1))
            fast_fut = fast_fut_input[:, :cutoff]

            # Store variables per sample for analysis
            for i in range(B):
                sample_idx = batch_idx * test_loader.batch_size + i
                idx.append(sample_idx)

                #### Metrics with buffer zone ####
                buffer_masks = buffer_mask(target_seq[i])
                dice_buffer = dice_score_shoreline_buffer(output[i], target_seq[i], buffer_masks, delta=0.5)

                dice_b += dice_buffer.sum().item()
                ious, recalls, precisions = buffer_zone_metrics(output[i], target_seq[i], buffer_masks)
                iou_s += ious.sum().item()
                precision_s += precisions.sum().item()
                recall_s += recalls.sum().item()

                # Loss per sample (only works if criterion2 returns per-sample loss)
                dice_scores.append(dice[i].mean().item())
                dice_buffer_s.append(dice_buffer.mean().item())
                iou_scores.append(ious.mean().item())
                precision_scores.append(precisions.mean().item())
                recall_scores.append(recalls.mean().item())

                #### Store variables per sample for analysis ###
                all_inputs.append(slow_input[i].cpu())              # shape: [T, 1, H, W]
                all_targets.append(target_seq[i].cpu())            # shape: [T, 1, H, W]
                all_outputs.append(output[i].cpu())                # shape: [T, 1, H, W]
                all_wave_contexts.append(fast_input[i].cpu())      # shape: [W_input_len, 3]
                all_wave_contexts_future.append(fast_fut[i].cpu())# shape: [W_future_len, 3]
                all_buffer_masks.append(buffer_masks.cpu())

            # Per-frame metrics
            for t in range(future_seq if future_seq > 1 else 1):
                pred_t = output[:, t] 
                target_t = target_seq[:, t] 

                # Compute buffer mask from true target
                buffer_masks = buffer_mask(target_t)

                # Compute buffer-aware metrics
                dice_buffer = dice_score_shoreline_buffer(pred_t, target_t, buffer_masks)
                ious, recalls, precisions = buffer_zone_metrics(pred_t, target_t, buffer_masks)

                # Store per-frame metrics
                per_frame_dice[t].extend(dice_score(pred_t, target_t).cpu().tolist())  # Full-image Dice
                per_frame_dice_buffer[t].extend(dice_buffer.cpu().tolist())            # Buffer Dice
                per_frame_iou[t].extend(ious.cpu().tolist())
                per_frame_precision[t].extend(precisions.cpu().tolist())
                per_frame_recall[t].extend(recalls.cpu().tolist())

    #### Run diagnostics and visualizations ####
    diagnose(all_outputs, all_targets, dice_buffer_s, all_inputs, all_wave_contexts, all_wave_contexts_future, title_prefix="Testing", future_seq=future_seq)
    buffer_zone_vis(all_targets, all_outputs, all_buffer_masks, title="Testing", future_seq=future_seq)

    #### AVG Metrics ####
    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_dice_buffer = sum(dice_buffer_s) / len(dice_buffer_s)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)

    avg_per_frame_dice = [np.mean(vals) for vals in per_frame_dice]
    avg_per_frame_dice_buffer = [np.mean(vals) for vals in per_frame_dice_buffer]
    avg_per_frame_iou = [np.mean(vals) for vals in per_frame_iou]
    avg_per_frame_precision = [np.mean(vals) for vals in per_frame_precision]
    avg_per_frame_recall = [np.mean(vals) for vals in per_frame_recall]


    test_data.extend([
    ["Dice"] + avg_per_frame_dice,
    ["Dice Buffer"] + avg_per_frame_dice_buffer,
    ["IOU"] + avg_per_frame_iou,
    ["Precision"] + avg_per_frame_precision,
    ["Recall"] + avg_per_frame_recall
])
    
    print("====> Testing Total samples:", total_samples)
    print('====> Testing set Dice: {:.4f}'.format(avg_dice))
    print('====> Testing set Dice Buffer: {:.4f}'.format(avg_dice_buffer))
    print('====> Testing set IOU: {:.4f}'.format(avg_iou))
    print('====> Testing set Precision: {:.4f}'.format(avg_precision))
    print('====> Testing set Recall: {:.4f}'.format(avg_recall))

    #### Log metrics to WandB per sample ####
    for i in range(len(idx)):
        wandb.log({
            "Dice_Score Testing": dice_scores[i],
            "Dice_Score Buffer Testing": dice_buffer_s[i],
            "IOU_Score Testing": iou_scores[i],
            "Precision_Score Testing": precision_scores[i],
            "Recall_Score Testing": recall_scores[i],
            "Sample index Testing": idx[i] + 1
        })