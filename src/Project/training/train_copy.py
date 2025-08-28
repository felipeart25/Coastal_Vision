import torch
import wandb
from utils.metrics import buffer_mask_vec


def train(model, train_loader, criterion, optimizer, device, epoch, future_seq=1):

    #### Set model to training mode ####
    model.train()
    train_loss = 0
    total_batches = len(train_loader)
    
    #### Track total time for the epoch
    start_epoch = torch.cuda.Event(enable_timing=True)
    end_epoch = torch.cuda.Event(enable_timing=True)
    start_epoch.record()

    #### Loop through training data per batch ####
    for batch_idx, (slow_input, target_seq, fast_input, fast_fut_input) in enumerate(train_loader):

        #### Move data to device ####
        slow_input = slow_input.to(device)
        fast_input = fast_input.to(device)
        target_seq = target_seq.to(device)
        fast_fut_input = fast_fut_input.to(device)

        #### Log the time taken for each iteration ####
        start_iter = torch.cuda.Event(enable_timing=True)
        end_iter = torch.cuda.Event(enable_timing=True)
        start_iter.record()

        #### zero gradients to avoid accumulation ####
        optimizer.zero_grad()

        #### Output from the model ####
        output = model(slow_input, fast_input, fast_fut_input, future_seq=future_seq)

        if future_seq == 1:
            target_seq = target_seq[:, 0]
        else:
            target_seq = target_seq[:, :future_seq]


        loss = criterion(output, target_seq)

        #### Compute gradients backward to learn ####
        loss.backward()

        #### Clip gradients to avoid exploding gradients ####
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #### Update weights ####
        optimizer.step()

        #### Log the time taken for each iteration ####
        end_iter.record()
        torch.cuda.synchronize()
        iteration_time = start_iter.elapsed_time(end_iter)/1000

        #### Store loss for the epoch ####
        train_loss += loss.item()
        
        # Log time to WandB
        wandb.log({"Batch Loss": loss.item(), "Iteration Time (s)": iteration_time})

        #### Print progress every 100 batches ####
        if batch_idx % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch}, Batch {batch_idx}, Time per Iteration: {iteration_time:.4f}s")

    #### time taken for the epoch ####
    end_epoch.record()
    torch.cuda.synchronize()
    epoch_time = start_epoch.elapsed_time(end_epoch)/1000
    avg_iteration_time = epoch_time / total_batches
    
    #### Average loss for the epoch ####
    avg_train_loss = train_loss / len(train_loader)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_train_loss),  )
    
    # Log epoch train loss to WandB
    wandb.log({"Epoch Train Loss": avg_train_loss, "Epoch": epoch, "Avg Iteration Time (s)": avg_iteration_time})
    
    return avg_train_loss