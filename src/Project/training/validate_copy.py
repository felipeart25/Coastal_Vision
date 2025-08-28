import torch
import numpy as np
import wandb
from utils.metrics import dice_score, dice_score_shoreline_buffer, buffer_zone_metrics, buffer_mask, buffer_mask_vec

def validate(model, val_loader, criterion, device, epoch, eval_table_data, future_seq=1):
    model.eval()

    #### Define variables to store #####
    per_sample_losses = []
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

    # Initialize per-frame metric lists
    per_frame_dice = [[] for _ in range(future_seq)]
    per_frame_dice_buffer = [[] for _ in range(future_seq)]
    per_frame_iou = [[] for _ in range(future_seq)]
    per_frame_precision = [[] for _ in range(future_seq)]
    per_frame_recall = [[] for _ in range(future_seq)]

    idx = []
    val_loss = 0
    total_samples = 0
    dice_s = 0
    dice_b = 0
    iou_s = 0
    precision_s = 0
    recall_s = 0

    with torch.no_grad():
        for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(val_loader):
            slow_input = slow_input.to(device)
            fast_input = fast_input.to(device)
            target_seq = target_seq.to(device)
            fast_fut_input = fast_fut_input.to(device)

            # Forward pass
            output = model(slow_input, fast_input, fast_fut_input, future_seq=future_seq)

            # Handle targets based on strategy
            if future_seq == 1:
                target_seq = target_seq[:, :future_seq]
            else:
                target_seq = target_seq[:, :future_seq]

            # Compute overall loss
            

            loss = criterion(output, target_seq)


            val_loss += loss.item()

            B = output.size(0)
            total_samples += B
            
            # Compute Dice over full prediction
            dice = dice_score(output, target_seq)
            dice_s += dice.sum().item()

            
            # Cut off future wave data (do this once per batch)
            cutoff = future_seq * (fast_fut_input.size(1) // target_seq.size(1))
            fast_fut = fast_fut_input[:, :cutoff]
            # Store variables per sample
            for i in range(B):
                sample_idx = batch_idx * val_loader.batch_size + i
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
                per_sample_losses.append(loss[i].item() if len(loss.shape) > 0 else loss.item())
                dice_scores.append(dice[i].mean().item())
                dice_buffer_s.append(dice_buffer.mean().item())
                iou_scores.append(ious.mean().item())
                precision_scores.append(precisions.mean().item())
                recall_scores.append(recalls.mean().item())

                # Save inputs/outputs
                all_inputs.append(slow_input[i].cpu())
                all_targets.append(target_seq[i].cpu())
                all_outputs.append(output[i].cpu())
                all_wave_contexts.append(fast_input[i].cpu())
                all_wave_contexts_future.append(fast_fut[i].cpu())
                all_buffer_masks.append(buffer_masks.cpu())
            
            # Compute per-frame metrics (optional, useful for analysis)
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

            

    #### AVG Loss and Metrics per Epoch ####
    avg_val_loss = val_loss / len(val_loader)
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

    print("====> Validation Total samples:", total_samples)
    print('====> Validation set loss: {:.4f}'.format(avg_val_loss))
    print('====> Validation set Dice: {:.4f}'.format(avg_dice))
    print('====> Validation set Dice Buffer: {:.4f}'.format(avg_dice_buffer))
    print('====> Validation set IOU: {:.4f}'.format(avg_iou))
    print('====> Validation set Precision: {:.4f}'.format(avg_precision))
    print('====> Validation set Recall: {:.4f}'.format(avg_recall))
    print('====> Validation set per-frame Dice:', avg_per_frame_dice_buffer)

    wandb.log({
        "Epoch Validation Loss": avg_val_loss,
        "Epoch Validation Dice": avg_dice,
        "Epoch Validation Dice Buffer": avg_dice_buffer,
        "Epoch Validation IOU": avg_iou,
        "Epoch Validation Precision": avg_precision,
        "Epoch Validation Recall": avg_recall,
        "Epoch": epoch
    })

    return (
        avg_recall,
        avg_precision,
        avg_iou,
        avg_dice_buffer,
        avg_val_loss,
        per_sample_losses,
        dice_buffer_s,
        idx,
        all_outputs,
        all_targets,
        all_inputs,
        all_wave_contexts,
        all_wave_contexts_future,
        all_buffer_masks
    )