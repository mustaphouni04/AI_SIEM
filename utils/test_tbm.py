import wandb
import torch

def test(model, test_loader, device="cuda", save=True):
    model.to(device)
    with torch.no_grad():
        mse_total = 0
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            mse = torch.mean((inputs - outputs)**2, dim=(1, 2))
            mse_total += mse.sum().item()

        avg_mse = mse_total / len(test_loader.dataset)
        print(f"Average MSE of the model on the test set: {avg_mse}")
        wandb.log({"test_avg_mse": avg_mse})

        dict_reconstructions = dict()
        # Calculate reconstruction loss for each sequence
        for i, inputs in enumerate(test_loader):
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            mse_per_seq = torch.mean((inputs - outputs)**2)  # Calculate mean over all time steps
            print(f"Reconstruction loss for sequence {i}: {mse_per_seq.item()}")
            dict_reconstructions[i] = mse_per_seq.item()


    if save:
        # Save the dictionary in a txt file
        with open("reconstructions.txt", "w") as f:
            for key, value in dict_reconstructions.items():
                f.write(f"Sequence {key}: {value}\n")
                
        # Save the model in the exchangeable ONNX format
        # Note: Adjusting input and output names for autoencoder
        dummy_input = torch.randn(1, 20, 25).to(device)  # Ensure dummy input is on the same device
        torch.onnx.export(model,  # model being run
                          dummy_input,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['reconstructed_output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                       'reconstructed_output': {0: 'batch_size'}})
        wandb.save("model.onnx")