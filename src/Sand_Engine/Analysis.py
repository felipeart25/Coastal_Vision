import numpy as np
import matplotlib.pyplot as plt
import wandb
import numpy.ma as ma
import torch
import os


def diagnose(outputs, targets, dice_scores, inputs, wave_inputs, wave_fut, title_prefix="", future_seq=1):

    # Convert to numpy arrays
    outputs = [o.numpy() if isinstance(o, torch.Tensor) else o for o in outputs]
    targets = [t.numpy() if isinstance(t, torch.Tensor) else t for t in targets]

    # Take only last prediction when future_seq > 1
    if future_seq > 1:
        outputs_last = [o[-1] for o in outputs]  # Only last predicted frame
        targets_last = [t[-1] for t in targets]  # Only last target
        inputs_last = [i[-1] for i in inputs]    # Or take full input sequence if needed
        dice_scores = np.array(dice_scores)       # Ensure it's aligned
    else:
        outputs_last = [o[0] for o in outputs]
        targets_last = [t[0] for t in targets]
        inputs_last = [i[0] for i in inputs] 

    # Convert to numpy for easy sorting
    scores = np.array(dice_scores)
    best_indices = np.argsort(scores)[-10:][::-1]  # 10 best
    worst_indices = np.argsort(scores)[:10]        # 10 worst

    def plot_samples(indices, tag):
        
        project_name = wandb.run.name
        folder_outputs = os.path.join("..", "..", "outputs", project_name)
        os.makedirs(folder_outputs, exist_ok=True)

        
        fig, axes = plt.subplots(3, 10, figsize=(40, 12))
        for i, idx in enumerate(indices):
            pred = outputs_last[idx][0]  # shape: [H, W]
            true = targets_last[idx][0]  # shape: [H, W]
            pred_bin = (pred > 0.5).astype(np.uint8)  # Binary prediction

            fp = (pred_bin == 1) & (true == 0)
            fn = (pred_bin == 0) & (true == 1)
            tp = (pred_bin == 1) & (true == 1)
            tn = (pred_bin == 0) & (true == 0)

            tp = tp.squeeze().astype(np.uint8)
            tn = tn.squeeze().astype(np.uint8)
            fp = fp.squeeze().astype(np.uint8)
            fn = fn.squeeze().astype(np.uint8)

            rgb = np.zeros((tp.shape[0], tp.shape[1], 3), dtype=np.float32)
            rgb[..., 0] = fn     # Red channel = FN
            rgb[..., 1] = fp     # Green channel = FP
            rgb[tp == 1] = [1.0, 1.0, 1.0]  # White color for TP

            axes[0, i].imshow(true.squeeze(), cmap='gray', origin='lower')
            axes[0, i].set_title(f'Ground Truth')
            axes[0, i].axis('off')

            axes[1, i].imshow(pred.squeeze(), cmap='gray', origin='lower')
            axes[1, i].set_title(f'Prediction')
            axes[1, i].axis('off')

            axes[2, i].imshow(rgb, origin='lower')
            axes[2, i].set_title(f'Idx={idx}\nRed=FN, Green=FP\nDice={scores[idx]:.2f}')
            axes[2, i].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(folder_outputs, f'{tag}_samples.png')
        plt.savefig(plot_path, dpi=300)
        wandb.log({f"{title_prefix} {tag} Samples": wandb.Image(plot_path)})
        plt.close()

    def visualize_dynamics(inputs, targets, predictions, wave_inputs, wave_fut, indices, tag):

        project_name = wandb.run.name
        folder_outputs = os.path.join("..", "..", "outputs", project_name)
        os.makedirs(folder_outputs, exist_ok=True)
        for i, idx in enumerate(indices):

            predictions = [o.numpy() if isinstance(o, torch.Tensor) else o for o in predictions]
            targets = [t.numpy() if isinstance(t, torch.Tensor) else t for t in targets]
            wave_fut = [w.numpy() if isinstance(w, torch.Tensor) else w for w in wave_fut]

            
            # Take only last prediction when future_seq > 1
            if future_seq > 1:
                outputs_last = [o[-1] for o in predictions]  # Only last predicted frame
                targets_last = [t[-1] for t in targets] 
                wave_fut = wave_fut     # Only last future wave
            else:
                outputs_last = [o[0] for o in predictions]
                targets_last = [t[0] for t in targets]
                wave_fut = wave_fut     # Only last future wave
                #inputs_last = [i[0] for i in inputs] 
            input_seq = inputs[idx].squeeze(1).numpy()     # shape (10, H, W)
            target = targets_last[idx].squeeze()        # shape (H, W)
            pred = outputs_last[idx].squeeze()       # shape (H, W)
            wave = wave_inputs[idx].numpy()                 # shape (480, 1) — or (T,)
            wave_future = wave_fut[idx]           # shape (480, 1) — or (T,)

            chunks = np.array_split(wave.squeeze(), 10)

            fig, axes = plt.subplots(4, 11, figsize=(30, 10),
                         gridspec_kw={'height_ratios': [1, 1, 1, 1],
                                      'width_ratios': [1]*11})
            fig.suptitle(f"Analysis — Index: {idx} — Dice: {dice_scores[idx]:.3f}", fontsize=16)

            fig.text(0.01, 0.90, "Inputs", va='center', rotation='vertical', fontsize=12)
            fig.text(0.01, 0.60, "Dynamics", va='center', rotation='vertical', fontsize=12)
            fig.text(0.01, 0.38, "Error Map", va='center', rotation='vertical', fontsize=12)

            ### Row 0: Input frames + target
            for t in range(10):
                axes[0, t].imshow(input_seq[t], cmap='gray', origin='lower')
                axes[0, t].set_title(f'Input t={t+1}')
                axes[0, t].axis('off')
            axes[0, 10].imshow(target, cmap='gray', origin='lower')
            axes[0, 10].set_title('Target')
            axes[0, 10].axis('off')

            ### Row 1: Differences (dynamics)

            for ax in axes[1]:
                ax.set_facecolor('#cccccc')

            for t in range(9):
                diff = input_seq[t+1] - input_seq[t]
                abs_diff = np.abs(diff)
                change_pixels = (abs_diff > 0.01).sum()
                total_pixels = diff.size
                pct_change = (change_pixels / total_pixels) * 100
                mean_change = abs_diff.mean()

                ax = axes[1, t]  
                ax.imshow(diff, cmap='bwr', origin='lower', vmin=-1, vmax=1)
                ax.axis('off')

                title_text = f"Dynamics {t+1}-{t+2}\n Red=Increase\nBlue=Decrease\nΔPx: {change_pixels}\nΔ%: {pct_change:.1f}%"
                ax.text(
                    0.5, -0.1, title_text,
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=12
                )
            ax = axes[1, 9]  
            axes[1, 9].set_facecolor('#cccccc')
            diff = target - input_seq[9] 
            abs_diff = np.abs(diff)
            change_pixels = (abs_diff > 0.01).sum()
            total_pixels = diff.size
            pct_change = (change_pixels / total_pixels) * 100
            mean_change = abs_diff.mean() 
            ax.imshow(diff, cmap='bwr', origin='lower', vmin=-1, vmax=1) 
            axes[1, 9].axis('off')
            title_text = f"Dynamics {10}-Target\n Red=Increase\nBlue=Decrease\nΔPx: {change_pixels}\nΔ%: {pct_change:.1f}%"
            ax.text(
                0.5, -0.1, title_text,
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=12
            )

            axes[1, 10].imshow(pred, cmap='gray', origin='lower')
            axes[1, 10].set_title('Prediction')
            axes[1, 10].axis('off')


            ### Row 2: Difference (FN, FP)
            pred_bin = (pred > 0.5).astype(np.uint8)
            target_bin = (target > 0.5).astype(np.uint8)
            fn = (target_bin == 1) & (pred_bin == 0)
            fp = (target_bin == 0) & (pred_bin == 1)
            tp = (pred_bin == 1) & (target_bin == 1)

            rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.float32)
            rgb[..., 0] = fn  # red = FN
            rgb[..., 1] = fp  # green = FP
            rgb[tp == 1] = [1.0, 1.0, 1.0]  # White color for TP

            for j in range(10):
                axes[2, j].axis('off')
            axes[2, 10].imshow(rgb, origin='lower')
            axes[2, 10].set_title('Difference \nRed=FN, Green=FP')
            axes[2, 10].axis('off')

            ### Row 3: Wave input
            for t in range(10):

                ax = axes[3, t]
                wave_sub = chunks[t]
                ax.plot(wave_sub[:,0], color='blue')
                ax.set_ylabel("Hs [m]")
                ax.set_xlabel("Time steps")
                ax.set_ylim( np.max(wave[:, 0]) * -1.1, np.max(wave[:, 0]) * 1.1)  # Adjust y-axis for visibility
                ax.set_xticks([])
                ax.set_title(f'Wave Hs (m)', fontsize=12)

            # Plot full future wave (Hs)
            ax = axes[3, 10]
            ax.plot(wave_future[:, 0], color='black', label='Future Hs')
            ax.set_ylabel("Hs [m]")
            ax.set_xlabel("Time steps")
            ax.set_ylim( np.max(wave[:, 0]) * -1.1, np.max(wave[:, 0]) * 1.1)
            ax.set_xticks([])
            ax.set_title('Future Hs', fontsize=12)



            

            ### Final layout
            plt.subplots_adjust(hspace=0.2, top=0.92, bottom=0.05)
            plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.3)
            filename = f'dynamics_{tag}_{idx}.png'
            plot_path = os.path.join(folder_outputs, filename)
            plt.savefig(plot_path, dpi=300)
            wandb.log({f"{title_prefix} Dynamics {tag}": wandb.Image(plot_path)})
            plt.close()

        # Dynamics visualization for worst and best samples
    visualize_dynamics(inputs, targets, outputs, wave_inputs, wave_fut, worst_indices[:10], tag="worst_top10")
    #visualize_dynamics(inputs, targets, outputs, wave_inputs, wave_fut, worst_indices[10:], tag="worst_next10")
    visualize_dynamics(inputs, targets, outputs, wave_inputs, wave_fut, best_indices[:10], tag="best_top10")
    #visualize_dynamics(inputs, targets, outputs, wave_inputs, wave_fut, best_indices[10:], tag="best_next10")



    plot_samples(worst_indices[:10], "worst")
    #plot_samples(worst_indices[10:], "worst")
    plot_samples(best_indices[:10], "best")
    #plot_samples(best_indices[10:], "best")


import matplotlib.pyplot as plt
import wandb
import numpy as np

def buffer_zone_vis(targets, predictions, buffer_masks, title="", group_size=10, future_seq=1):
    """
    Logs batches of sample visualizations to wandb.
    Each figure contains up to `group_size` samples in rows with:
    - Ground truth
    - Prediction
    - Buffer zone overlay

    Args:
        targets: array-like, shape (B, H, W)
        predictions: array-like, shape (B, H, W)
        buffer_masks: array-like, shape (B, H, W)
        group_size: int, number of samples per figure
    """
    def to_numpy(x):
        if isinstance(x, list):
            x = torch.stack(x)
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.squeeze(x)

    

    predictions = to_numpy(predictions)  # (B, H, W)
    targets = to_numpy(targets)          # (B, 1, H, W) → we’ll squeeze the channel later
    buffer_masks = to_numpy(buffer_masks)  # (B, H, W)

    # Only now squeeze channel 1 for targets (keep B, H, W)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]

    # Take only last prediction when future_seq > 1
    if future_seq > 1:
        predictions = [o[-1] for o in predictions]  # Only last predicted frame
        targets = [t[-1] for t in targets]   # Or take full input sequence if needed
        buffer_masks = [t[-1] for t in buffer_masks] 
    else:
        predictions =  predictions
        targets = targets
        buffer_masks =  buffer_masks
    

     

    num_samples = len(buffer_masks)
    num_groups = (num_samples + group_size - 1) // group_size  # ceil division


    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, num_samples)


        
        
        batch_targets = targets[start:end]
        batch_preds = predictions[start:end]
        batch_buffers = buffer_masks[start:end]

        fig, axs = plt.subplots(len(batch_targets), 3, figsize=(15, 3 * len(batch_targets)))

        if len(batch_targets) == 1:
            axs = axs[np.newaxis, :]  # Ensure 2D array for consistent indexing

        for i in range(len(batch_targets)):
            global_idx = start + i  # Absolute index across the full dataset

            axs[i, 0].imshow(batch_targets[i], cmap='gray', origin='lower')
            axs[i, 0].set_title(f"Target #{global_idx}")
            axs[i, 0].axis('off')

            axs[i, 1].imshow(batch_preds[i], cmap='gray', origin='lower')
            axs[i, 1].set_title(f"Prediction #{global_idx}")
            axs[i, 1].axis('off')

            axs[i, 2].imshow(batch_targets[i], cmap='gray', origin='lower', alpha=0.5)
            axs[i, 2].imshow(batch_buffers[i], cmap='Reds', origin='lower', alpha=0.5)
            axs[i, 2].set_title(f"Buffer Zone #{global_idx}")
            axs[i, 2].axis('off')

        plt.tight_layout()
        wandb.log({f"{title} Shoreline Evaluation": wandb.Image(fig)})
        plt.close()

import numpy as np
import matplotlib.pyplot as plt

def Inference_analysis(targets, predictions, dice_scores, title="", group_size=10, future_seq=50):
    """
    targets: Tensor [N, T, H, W]
    predictions: Tensor [N, T, H, W]
    dice_scores: list or array of length N
    """
    if isinstance(targets, list):
        targets = torch.stack(targets)
    if isinstance(predictions, list):
        predictions = torch.stack(predictions)

    # Convert to numpy
    targets = targets.cpu().numpy().squeeze()  # (N, T, H, W) → (N, T, H, W)
    print(f"Targets shape: {targets.shape}")
    predictions = predictions.cpu().numpy().squeeze()  # (N, T, H, W) → (N, T, H, W)
    print(f"Predictions shape: {predictions.shape}")
    dice_scores = np.array(dice_scores)
    print(f"Dice scores shape: {dice_scores.shape}")

    # Get indices of best and worst predictions
    best_indices = np.argsort(dice_scores)[-group_size:]  # Highest dice
    print(f"Best indices: {best_indices}")
    worst_indices = np.argsort(dice_scores)[:group_size]  # Lowest dice
    print(f"Worst indices: {worst_indices}")

    def plot_group(indices, group_title):
        num_samples = len(indices)
        print(f"Number of samples in group '{group_title}': {num_samples}")
        selected_frames = list(range(0, future_seq, future_seq // 10))  # every 5 if future_seq=50
        print(f"Selected frames: {selected_frames}")

        

        for index in indices:
            print(f"Processing index {index} for group '{group_title}'")

            fig, axs = plt.subplots(len(selected_frames), 3, figsize=(12, 3 * len(selected_frames)))
            fig.suptitle(f"Sample #{index} – Dice Score: {dice_scores[index]:.3f}", fontsize=16)
            
            for row_idx, t in enumerate(selected_frames):

                print(f"Processing index {index} for time step {t}")

                target = targets[index, t]
                pred = predictions[index, t]

                # Binarize
                target_bin = (target > 0.5).astype(np.uint8)
                pred_bin = (pred > 0.5).astype(np.uint8)

                # FN, FP, TP
                fn = (target_bin == 1) & (pred_bin == 0)
                fp = (target_bin == 0) & (pred_bin == 1)
                tp = (pred_bin == 1) & (target_bin == 1)

                # RGB difference image
                rgb = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.float32)
                rgb[..., 0] = fn  # Red: FN
                rgb[..., 1] = fp  # Green: FP
                rgb[tp == 1] = [1.0, 1.0, 1.0]  # White: TP

                axs[row_idx, 0].imshow(target_bin, cmap='gray', origin='lower')
                axs[row_idx, 0].set_title(f"Target t={t}")
                axs[row_idx, 0].axis('off')

                axs[row_idx, 1].imshow(pred_bin, cmap='gray', origin='lower')
                axs[row_idx, 1].set_title(f"Prediction t={t}")
                axs[row_idx, 1].axis('off')

                axs[row_idx, 2].imshow(rgb, origin='lower')
                axs[row_idx, 2].set_title(f"Difference\nRed=FN, Green=FP, White=TP")
                axs[row_idx, 2].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            wandb.log({f"{group_title} - {title}": wandb.Image(fig)})
            plt.close()
        

    # Plot best
    plot_group(best_indices, "Top 10 Dice Scores")

    # Plot worst
    plot_group(worst_indices, "Worst 10 Dice Scores")

