import torch
from models.ConvLSTMabla import Predictor 
from data.data_loader_v3 import test_loader_inference2  # your DataLoader for test data
from utils.metrics import dice_score, dice_score_shoreline_buffer, buffer_mask, buffer_zone_metrics, compute_area_changes  # your metric functions
from utils.Analysis import Inference_analysis_detailed # your visualization function
from utils.shoreline_analysis import shoreline_displacement_analysis, extract_shoreline, compute_normals, create_transects, plot_transects_and_shorelines, resample_shoreline
import os
import wandb
import numpy as np
import pandas as pd

def load_model(model_path, device, future_seq=1):
    # Reconstruct the model with the same architecture used during training
    model = Predictor(
        input_dim=1,
        hidden_dims=[64, 64],
        kernel_sizes=[(5, 5), (5, 5)],
        num_layers=2,
        lstm_hidden_size=64,
        dropout=0.2,
        wave_steps_per_frame=36,
        future_seq=future_seq,
        use_wave_context=True  # Set to True if your model uses wave context
    )
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        # Remove 'module.' from keys
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def run_inference(model, test_loader, device, inference_data, future_seq=50, results_dir="results"):
    model.eval()

    all_inputs = []
    all_targets = []
    all_outputs = []
    all_wave_contexts = []
    all_wave_contexts_future = []
    all_buffer_masks = []

    overpredicted_areas_per_sample = []  # shape: [n_samples, future_seq]
    underpredicted_areas_per_sample = []    # same shape
    land_areas_per_sample = []  # shape: [n_samples, future_seq]
    relative_overpredicted_per_sample = []  # shape: [n_samples, future_seq]
    relative_underpredicted_per_sample = []  # shape: [n_samples, future_seq]
    shoreline_displacement_per_sample = []  # shape: [n_samples, future_seq]
    transect_shifts_accum = [] 

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

                overpredicted_seq = []
                underpredicted_seq = []
                land_areas_seq = []
                relative_overpredicted_seq = []
                relative_underpredicted_seq = []
                displacement_seq = []
                transect_shifts_accum_seq = []  # Reset for each sample

            for t in range(future_seq if future_seq > 1 else 1):
                
                pred_t = output[:, t] 
                target_t = target_seq[:, t] 

                shift_values = shoreline_displacement_analysis(target_t, pred_t) 
                mean_shift = np.nanmean(shift_values)
                displacement_seq.append(mean_shift)
                transect_shifts_accum_seq.append(shift_values)  # Store all shifts for detailed analysis
                

                # Compute buffer mask from true target
                buffer_masks = buffer_mask(target_t)

                # Compute buffer-aware metrics
                dice_buffer = dice_score_shoreline_buffer(pred_t, target_t, buffer_masks)
                ious, recalls, precisions = buffer_zone_metrics(pred_t, target_t, buffer_masks)

                over, under, land_area, relative_over, relative_under = compute_area_changes(pred_t, target_t)
                overpredicted_seq.append(over)
                underpredicted_seq.append(under)
                land_areas_seq.append(land_area)
                relative_overpredicted_seq.append(relative_over)
                relative_underpredicted_seq.append(relative_under)

                wandb.log({"Dice Buffer Along all test samples": dice_buffer,
                            "IoU Buffer Along all test samples": ious,
                            "Precision Buffer Along all test samples": precisions,
                            "Recall Buffer Along all test samples": recalls})
                

                # Store per-frame metrics
                per_frame_dice[t].extend(dice_score(pred_t, target_t).cpu().tolist())  # Full-image Dice
                per_frame_dice_buffer[t].extend(dice_buffer.cpu().tolist())            # Buffer Dice
                per_frame_iou[t].extend(ious.cpu().tolist())
                per_frame_precision[t].extend(precisions.cpu().tolist())
                per_frame_recall[t].extend(recalls.cpu().tolist())

            overpredicted_areas_per_sample.append(overpredicted_seq)  # shape: [n_samples, future_seq]
            underpredicted_areas_per_sample.append(underpredicted_seq)      # same shape
            land_areas_per_sample.append(land_areas_seq)                    # shape: [n_samples, future_seq]
            relative_overpredicted_per_sample.append(relative_overpredicted_seq)  # shape: [n_samples, future_seq]
            relative_underpredicted_per_sample.append(relative_underpredicted_seq)  # shape: [n_samples, future_seq]
            shoreline_displacement_per_sample.append(displacement_seq)
            transect_shifts_accum.append(transect_shifts_accum_seq)

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

    #Inference_analysis(all_targets, all_outputs, dice_buffer_s, title='Inference', future_seq=future_seq)  
    Inference_analysis_detailed(all_targets, all_outputs, dice_buffer_s, title='Inference', future_seq=future_seq) 

    # Convert list to array: shape [n_samples * T, n_transects]
    #transect_shifts_accum = np.array(transect_shifts_accum) 

    # Now compute mean, min, max per transect
    #mean_shift_per_transect = np.nanmean(transect_shifts_accum, axis=0)
    #min_shift_per_transect = np.nanmin(transect_shifts_accum, axis=0)
    #max_shift_per_transect = np.nanmax(transect_shifts_accum, axis=0)
    #std_shift_per_transect = np.nanstd(transect_shifts_accum, axis=0)

    # Optionally, save for future plots
    #np.save(f"{results_dir}/mean_shift_per_transect.npy", mean_shift_per_transect)
    #np.save(f"{results_dir}/min_shift_per_transect.npy", min_shift_per_transect)
    #np.save(f"{results_dir}/max_shift_per_transect.npy", max_shift_per_transect)
    #np.save(f"{results_dir}/std_shift_per_transect.npy", std_shift_per_transect)

    # n_transects = len(mean_shift_per_transect)
    # df_transects = pd.DataFrame({
    #     "Transect": np.arange(1, n_transects + 1),
    #     "Mean Shift (m)": mean_shift_per_transect,
    #     "Min Shift (m)": min_shift_per_transect,
    #     "Max Shift (m)": max_shift_per_transect,
    #     "Std Shift (m)": std_shift_per_transect
    # })

    #wandb.log({"Shoreline Shift per Transect": wandb.Table(dataframe=df_transects)})

    overpredicted_areas_per_sample = np.array(overpredicted_areas_per_sample)  # shape: [n_samples, T]
    underpredicted_areas_per_sample = np.array(underpredicted_areas_per_sample)
    land_areas_per_sample = np.array(land_areas_per_sample)
    relative_overpredicted_per_sample = np.array(relative_overpredicted_per_sample)
    relative_underpredicted_per_sample = np.array(relative_underpredicted_per_sample)
    shoreline_displacement_per_sample = np.array(shoreline_displacement_per_sample)

    # Mean per sample
    mean_overpredicted_per_sample = overpredicted_areas_per_sample.mean(axis=1)
    mean_underpredicted_per_sample = underpredicted_areas_per_sample.mean(axis=1)
    mean_land_area_per_sample = land_areas_per_sample.mean(axis=1)
    mean_relative_overpredicted_per_sample = relative_overpredicted_per_sample.mean(axis=1)
    mean_relative_underpredicted_per_sample = relative_underpredicted_per_sample.mean(axis=1)
    mean_shoreline_displacement_per_sample = shoreline_displacement_per_sample.mean(axis=1)

    # Log to wandb
    for sample_idx in range(len(mean_overpredicted_per_sample)):
        wandb.log({
            "Mean Overpredicted Area [km²] per Sample": mean_overpredicted_per_sample[sample_idx],
            "Mean Underpredicted Area [km²] per Sample": mean_underpredicted_per_sample[sample_idx],
            "Mean Land Area [km²] per Sample": mean_land_area_per_sample[sample_idx],
            "Mean Relative Overpredicted Area [%] per Sample": mean_relative_overpredicted_per_sample[sample_idx],
            "Mean Relative Underpredicted Area [%] per Sample": mean_relative_underpredicted_per_sample[sample_idx],
            "Mean Shoreline Displacement [m] per Sample": mean_shoreline_displacement_per_sample[sample_idx],
            "Sample Index": sample_idx + 1
        })

    # Mean per frame
    mean_overpredicted_per_frame = overpredicted_areas_per_sample.mean(axis=0)
    mean_underpredicted_per_frame = underpredicted_areas_per_sample.mean(axis=0)
    mean_land_area_per_frame = land_areas_per_sample.mean(axis=0)
    mean_relative_overpredicted_per_frame = relative_overpredicted_per_sample.mean(axis=0)
    mean_relative_underpredicted_per_frame = relative_underpredicted_per_sample.mean(axis=0)
    mean_shoreline_displacement_per_frame = shoreline_displacement_per_sample.mean(axis=0)

    # Log to wandb
    for t in range(future_seq):
        wandb.log({
            "Mean Overpredicted Area [km²] per Frame": mean_overpredicted_per_frame[t],
            "Mean Underpredicted Area [km²] per Frame": mean_underpredicted_per_frame[t],
            "Mean Land Area [km²] per Frame": mean_land_area_per_frame[t],
            "Mean Relative Overpredicted Area [%] per Frame": mean_relative_overpredicted_per_frame[t],
            "Mean Relative Underpredicted Area [%] per Frame": mean_relative_underpredicted_per_frame[t],
            "Mean Shoreline Displacement [m] per Frame": mean_shoreline_displacement_per_frame[t],
            "Frame Index": t + 1
        })

    

    # Store for plots or further analysis
    np.save(f"{results_dir}/overpredicted_areas_per_sample.npy", overpredicted_areas_per_sample)
    np.save(f"{results_dir}/underpredicted_areas_per_sample.npy", underpredicted_areas_per_sample)
    np.save(f"{results_dir}/relative_overpredicted_per_sample.npy", relative_overpredicted_per_sample)
    np.save(f"{results_dir}/relative_underpredicted_per_sample.npy", relative_underpredicted_per_sample)
    np.save(f"{results_dir}/land_areas_per_sample.npy", land_areas_per_sample)
    np.save(f"{results_dir}/dice_buffer_s.npy", dice_buffer_s)
    np.save(f"{results_dir}/shoreline_displacement_per_sample.npy", shoreline_displacement_per_sample)
    #np.save(f"{results_dir}/transect_shifts_accum.npy", transect_shifts_accum)

    import random
    import os

    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot random N samples (e.g. 2)
    n_samples_to_plot = 2
    sample_indices = random.sample(range(len(all_targets)), min(n_samples_to_plot, len(all_targets)))

    for i in sample_indices:
        target_seq = all_targets[i]  # shape: [T, 1, H, W]
        pred_seq = all_outputs[i]

        for t in [0, len(target_seq) - 1]:  # First and last time step
            gt_mask = target_seq[t]
            pred_mask = pred_seq[t]

            gt_shoreline = extract_shoreline(gt_mask)
            if gt_shoreline is None:
                print(f"[Warning] No shoreline found in GT for sample {i}, frame {t}")
                continue
            gt_shoreline = resample_shoreline(gt_shoreline, spacing=5.0)        
            try:
                normals = compute_normals(gt_shoreline)
                transects = create_transects(gt_shoreline, normals)
                plot_transects_and_shorelines(
                    gt_mask,
                    pred_mask,
                    transects,
                    results_dir=plot_dir,
                    sample_idx=f"{i}_t{t}",
                    title="Shoreline and Transects"
                )
            except Exception as e:
                print(f"[Error] Could not plot sample {i}, frame {t}: {e}")




def main():

    wandb.login(key="d42992a374fbc96ee65d1955f037e71d58e30f45")
    wandb.init(project="THESIS",
        name=f"test",
    )
    exp_name = wandb.run.name
    results_folder = f"/p/11207608-coclico/MSc_students/Daniel/Scripts/outputs/Results/exp1/{exp_name}/"
    os.makedirs(results_folder, exist_ok=True)

    model_path = "/u/arteagag/Coastal_Vision/outputs/hybrid_v14_30Epochs_case1_biweekly_10fut/best_model.pth"  # Path to your saved checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    future_seq = 30
    batch_size = 1
    time = "bi-monthly"  # or "weekly", "monthly"
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
    run_inference(model, test_loader, device, inference_data, future_seq, results_folder)

    table = ["Metric"] + [f"t={i+1}" for i in range(future_seq)]
    df = pd.DataFrame(inference_data, columns=table)

    wandb.log({"Inference Metrics": wandb.Table(dataframe=df)})
    # Save predictions if needed
    os.makedirs("results", exist_ok=True)
    #torch.save(predictions, "results/test_predictions.pt")
    #print(f"✅ Inference done! Predictions shape: {predictions.shape}")

if __name__ == "__main__":
    main()
