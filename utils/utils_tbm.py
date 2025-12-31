import wandb
import torch
import torch.nn 
from models.model_tbm import *
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def get_data_from_csv(csv_path, train=True, slice=1, subset_size=None):
    # read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    grouped = df.groupby('remote_host')
    remote_hosts_ordered = [remote_host for remote_host, _ in grouped]

    max_timesteps = 20
    sequences = []
    for remote_host, group_data in grouped:
        # drop 'remote_host' column as it's no longer needed for individual sequences
        group_data = group_data.drop(columns=['remote_host'])
        
        # convert the DataFrame to numpy array
        sequence_array = group_data.to_numpy()
        
        # pad the sequence with zeros if it's shorter than max_timesteps
        padded_sequence = pad_sequences([sequence_array], maxlen=max_timesteps, dtype='float32', padding='post', truncating='post')
        
        # append the padded sequence to the list of sequences
        sequences.append(padded_sequence)
    
    # stack all sequences to create the final input array for the LSTM
    X = np.vstack(sequences)
    # convert numpy array to PyTorch tensor
    X_tensor = torch.from_numpy(X)
    # if subset_size is specified, slice the tensor
    if subset_size is not None:
        X_tensor = X_tensor[:subset_size]

    # Split the data into training and testing sets based on the 'train' parameter
    split_index = int(0.8 * len(X_tensor))  # 80% for training
    if train:
        return X_tensor[:split_index], remote_hosts_ordered[:split_index]
    else:
        return X_tensor[split_index:], remote_hosts_ordered[split_index:]


def make_loader(dataset, batch_size):
    dataset = TensorDataset(dataset)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size, 
                        shuffle=False, # we don't want to shuffle, that is to keep order and get the correspondant anomalous IPs
                        pin_memory=True, num_workers=2)
    return loader


def make(config, csv_path, device="cuda"):
    # Make the data
    train_data, _ = get_data_from_csv(csv_path, train=True, subset_size=config.subset_size)
    test_data, _ = get_data_from_csv(csv_path, train=False, subset_size=config.subset_size)
    train_loader = make_loader(train_data, batch_size=config.batch_size)
    test_loader = make_loader(test_data, batch_size=config.batch_size)

    # Calculate input_dim based on the shape of the training data
    input_dim = train_data.shape[2]  # train_data is a 3D tensor
    
    hidden_dim = 100
    num_layers = 2
    
    model = LSTM_Autoencoder(input_dim, hidden_dim, num_layers).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    return model, train_loader, test_loader, criterion, optimizer

def get_test_remote_hosts(csv_path):
    test_data, remote_hosts = get_data_from_csv(csv_path, train=False)
    return remote_hosts