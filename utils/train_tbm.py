from tqdm.auto import tqdm
import wandb
import torch

def train(model, loader, criterion, optimizer, config):
    # Initialize Weights & Biases tracking
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Prepare for training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0

    # Training loop
    for epoch in tqdm(range(config.epochs)):
        for _, inputs in enumerate(loader):  # Only inputs are needed

            # Call the modified train_batch function
            loss = train_batch(inputs, model, optimizer, criterion)
            
            example_ct += len(inputs)
            batch_ct += 1

            # Log progress every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(inputs, model, optimizer, criterion, device="cuda"):
    # Move inputs to the specified device
    inputs = inputs[0].to(device)
    #tensor = torch.tensor(inputs).to(device)
    # Forward pass ➡
    outputs = model(inputs)
    loss = criterion(outputs, inputs)  # Use inputs as targets
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss.item()  # Return the loss value


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")