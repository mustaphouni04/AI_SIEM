import torch
import torch.nn as nn
import torch.optim as optim

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTM_Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Forward pass through LSTM
        out, _ = self.lstm(x)
        
        # Decode the output of the LSTM
        decoded = self.decoder(out)
        
        return decoded