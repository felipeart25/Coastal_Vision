import torch
from hybrid_v14_F import Predictor  # make sure Predictor is defined or imported
from data_loader_v3 import test_loader_inference2  # your DataLoader for test data
from metrics import dice_score, dice_score_shoreline_buffer, buffer_mask, buffer_zone_metrics  # your metric functions
from Analysis import buffer_zone_vis, Inference_analysis # your visualization function
import os
import wandb
import numpy as np
import pandas as pd

def load_model(model_path, device, future_seq=1):
    # Reconstruct the model with the same architecture used during training
    model = Predictor(
        input_dim=1,
        hidden_dims=[ 64, 64],
        kernel_sizes=[(5, 5), (5, 5)],
        num_layers=2,
        lstm_hidden_size=64,
        dropout=0.2,
        wave_steps_per_frame=36,
        future_seq=future_seq,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference(model, test_loader, device, inference_data, future_seq=50):
    model.eval()

    all_inputs = []
    all_targets = []
    all_outputs = []
    all_wave_contexts = []
    all_wave_contexts_future = []
    all_buffer_masks = []

    dice_buffer_s = []

    idx = []

    total_samples = 0

    # Initialize per-frame metric lists
    per_frame_dice = [[] for _ in range(future_seq)]
    per_frame_dice_buffer = [[] for _ in range(future_seq)]
    per_frame_iou = [[] for _ in range(future_seq)]
    per_frame_precision = [[] for _ in range(future_seq)]
    per_frame_recall = [[] for _ in range(future_seq)]


    with torch.no_grad():
        for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(test_loader):
            slow_input = slow_input.to(device)
            target_seq = target_seq.to(device)
            fast_input = fast_input.to(device)
            fast_fut_input = fast_fut_input.to(device)

            output = model(slow_input, fast_input, fast_fut_input, future_seq=future_seq, inference=True)   

            print(f"Output shape: {output.shape}")
            T = output.shape[1] 
            B = output.shape[0]  # Batch size
            total_samples += T
            print(f"Batch {batch_idx + 1}/{len(test_loader)}: {T} samples processed.")

            # Cut off future wave data
            cutoff = future_seq * (fast_fut_input.size(1) // target_seq.size(1))
            fast_fut = fast_fut_input[:, :cutoff]

            for i in range(B):
                sample_idx = batch_idx * test_loader.batch_size + i
                idx.append(sample_idx)

                #### Metrics with buffer zone ####
                buffer_masks = buffer_mask(target_seq[i])
                dice_buffer = dice_score_shoreline_buffer(output[i], target_seq[i], buffer_masks, delta=0.5)

                # Loss per sample (only works if criterion2 returns per-sample loss)
                dice_buffer_s.append(dice_buffer.mean().item())

                #### Store variables per sample for analysis ###
                all_inputs.append(slow_input[i].cpu())              # shape: [T, 1, H, W]
                all_targets.append(target_seq[i].cpu())            # shape: [T, 1, H, W]
                all_outputs.append(output[i].cpu())                # shape: [T, 1, H, W]
                all_wave_contexts.append(fast_input[i].cpu())      # shape: [W_input_len, 3]
                all_wave_contexts_future.append(fast_fut[i].cpu())# shape: [W_future_len, 3]
                all_buffer_masks.append(buffer_masks.cpu())

            for t in range(future_seq if future_seq > 1 else 1):
                
                pred_t = output[:, t] 
                target_t = target_seq[:, t] 

                # Compute buffer mask from true target
                buffer_masks = buffer_mask(target_t)

                # Compute buffer-aware metrics
                dice_buffer = dice_score_shoreline_buffer(pred_t, target_t, buffer_masks)
                ious, recalls, precisions = buffer_zone_metrics(pred_t, target_t, buffer_masks)

                wandb.log({"Dice Buffer Along all test samples": dice_buffer})
                # Store per-frame metrics
                per_frame_dice[t].extend(dice_score(pred_t, target_t).cpu().tolist())  # Full-image Dice
                per_frame_dice_buffer[t].extend(dice_buffer.cpu().tolist())            # Buffer Dice
                per_frame_iou[t].extend(ious.cpu().tolist())
                per_frame_precision[t].extend(precisions.cpu().tolist())
                per_frame_recall[t].extend(recalls.cpu().tolist())

    avg_per_frame_dice = [np.mean(vals) for vals in per_frame_dice]
    avg_per_frame_dice_buffer = [np.mean(vals) for vals in per_frame_dice_buffer]
    avg_per_frame_iou = [np.mean(vals) for vals in per_frame_iou]
    avg_per_frame_precision = [np.mean(vals) for vals in per_frame_precision]
    avg_per_frame_recall = [np.mean(vals) for vals in per_frame_recall]


    inference_data.extend([
    ["Dice"] + avg_per_frame_dice,
    ["Dice Buffer"] + avg_per_frame_dice_buffer,
    ["IOU"] + avg_per_frame_iou,
    ["Precision"] + avg_per_frame_precision,
    ["Recall"] + avg_per_frame_recall
])
    

    for t in range(future_seq):
        Dice = avg_per_frame_dice[t]
        Dice_Buffer = avg_per_frame_dice_buffer[t]
        IoU = avg_per_frame_iou[t]
        Precision = avg_per_frame_precision[t]
        Recall = avg_per_frame_recall[t]
        wandb.log({
            f"Dice": Dice,
            f"Dice Buffer": Dice_Buffer,
            f"IoU": IoU,
            f"Precision": Precision,
            f"Recall": Recall,
            f"Sample Index Inference": t + 1
        })

    Inference_analysis(all_targets, all_outputs, dice_buffer_s, title='Inference', future_seq=future_seq)  

    #all_outputs = (all_outputs > 0.5).float()
    #buffer_zone_vis(all_targets, all_outputs, all_buffer_masks, title="Inference")
    # Compute average metrics
    #avg_dice = np.mean(dice_scores)
    #avg_dice_buffer = np.mean(dice_buffer_s)
    #avg_iou = np.mean(iou_scores)
    #avg_precision = np.mean(precision_scores)
    #avg_recall = np.mean(recall_scores)

    #print(f"✅ Inference completed on {total_samples} samples.")
    #print(f"🔹 Dice: {avg_dice:.4f}")
    #print(f"🔹 Dice (Buffer Zone): {avg_dice_buffer:.4f}")
    #print(f"🔹 IoU (Buffer): {avg_iou:.4f}")
    #print(f"🔹 Precision (Buffer): {avg_precision:.4f}")
    #print(f"🔹 Recall (Buffer): {avg_recall:.4f}")

    # Log metrics per sample
    #for i in range(len(idx)):
    #    wandb.log({
    #        "Dice_Score Testing": dice_scores[i],
    #        "Dice_Score Buffer Testing": dice_buffer_s[i],
    #        "IOU_Score Testing": iou_scores[i],
    #        "Precision_Score Testing": precision_scores[i],
    #        "Recall_Score Testing": recall_scores[i],
    #        "Sample index Testing": idx[i] + 1
    #    }, step=idx[i] + 1)

    # Return concatenated output
    #return all_outputs  # (N, 1, H, W)

def main():

    wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")
    wandb.init(project="THESIS",
        name=f"Inference_hybrid_v14_30Epochs_case1_biweekly_10fut",
    )

    model_path = "/u/arteagag/Coastal_Vision/outputs/hybrid_v14_30Epochs_case1_biweekly_10fut/best_model.pth"  # Path to your saved checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    future_seq = 50
    batch_size = 1
    time = "bi-monthly"
    test_folder = f"/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/Inference/{time}/"
    time_dict = {
        "weekly": 34,
        "bi-monthly": 72,
        "monthly": 144
    }
    output_freq = 2  # hours
    time_key = time  # or "weekly", "monthly"
    wave_steps_per_frame = time_dict[time_key] // output_freq 

    # Load test data
    test_loader = test_loader_inference2(test_folder, future_seq, batch_size, time)

    # Load model
    model = load_model(model_path, device)
    print(model.wave_encoder.future_seq)
    model.wave_encoder.future_seq = future_seq
    model.wave_encoder.wave_steps_per_frame = wave_steps_per_frame
    print(f"Wave steps per frame: {model.wave_encoder.wave_steps_per_frame}")
    print(model.wave_encoder.future_seq)

    # Run inference
    inference_data = []
    run_inference(model, test_loader, device, inference_data, future_seq)

    table = ["Metric"] + [f"t={i+1}" for i in range(future_seq)]
    df = pd.DataFrame(inference_data, columns=table)
    wandb.log({"Inference Metrics": wandb.Table(dataframe=df)})
    # Save predictions if needed
    os.makedirs("results", exist_ok=True)
    #torch.save(predictions, "results/test_predictions.pt")
    #print(f"✅ Inference done! Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    main()
