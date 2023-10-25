import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tslearn.metrics import dtw
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from tqdm import tqdm

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerClassifier, self).__init__()
        print("input_dim",input_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4),
            num_layers=4
        )
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.classifier(x.mean(dim=1))
        return x

def sequence_classifier(real_data, synth_data, index_column, target_column, desired_seq_length=None):
    # Combine real and synthetic data
    data = pd.concat([real_data, synth_data])
    
    # Split data into sequences
    sequences = [group.drop(columns=[index_column, target_column]) for _, group in data.groupby(index_column)]
    lengths = [len(seq) for seq in sequences]
    labels = [group[target_column].iloc[0] for _, group in data.groupby(index_column)]
    
    # Convert sequences to tensors and pad them to desired sequence length
    padded_sequences = [torch.tensor(seq.values, dtype=torch.float32) for seq in sequences]
    if desired_seq_length:
        print("desired_seq_length is:",desired_seq_length)
        padded_sequences = [F.pad(seq, (0, 0, 0, desired_seq_length - seq.size(0))) if seq.size(0) < desired_seq_length else seq[:desired_seq_length] for seq in padded_sequences]
    sequences_tensor = torch.stack(padded_sequences)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Split data into training and testing sets
    train_sequences, test_sequences, train_labels, test_labels, train_lengths, test_lengths = train_test_split(
        sequences_tensor, labels_tensor, lengths, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Convert lengths to tensors
    train_lengths_tensor = torch.tensor(train_lengths, dtype=torch.int64)
    test_lengths_tensor = torch.tensor(test_lengths, dtype=torch.int64)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_sequences, train_labels, train_lengths_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_sequences, test_labels, test_lengths_tensor), batch_size=32, shuffle=False)

    
    # Initialize model, loss function, and optimizer
    model = TransformerClassifier(input_dim=train_sequences.size(2), num_classes=len(torch.unique(labels_tensor)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    for epoch in range(10):  # Assume 10 epochs for simplicity
        model.train()
        for batch_sequences, batch_labels, batch_lengths in train_loader:
            optimizer.zero_grad()
            # Pack the sequences
            packed_sequences = pack_padded_sequence(batch_sequences, batch_lengths, batch_first=True, enforce_sorted=False)
            outputs = model(packed_sequences)
            # Unpack the sequences if needed
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        all_preds = []
        for batch_sequences, _, batch_lengths in test_loader:
            # Pack the sequences
            packed_sequences = pack_padded_sequence(batch_sequences, batch_lengths, batch_first=True, enforce_sorted=False)
            outputs = model(packed_sequences)
            # Unpack the sequences if needed
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels.cpu().numpy(), all_preds)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy


def mmd_dtw(real_data, synth_data, index_column):
    # Split data frames into sequences based on unique values in the index column
    real_sequences = [group.drop(columns=index_column) for _, group in real_data.groupby(index_column)]
    synth_sequences = [group.drop(columns=index_column) for _, group in synth_data.groupby(index_column)]
    
    # Initialize one-hot encoder
    encoder = OneHotEncoder()
    
    # Initialize a variable to store the total MMD score
    total_mmd = 0
    
    # Number of pairs
    num_pairs = 0
    
    # Loop through each pair of real and synthetic sequences
    for real_seq in tqdm(real_sequences):
        for synth_seq in synth_sequences:
            # One-hot encode the sequences
            real_encoded = encoder.fit_transform(real_seq).toarray()
            synth_encoded = encoder.fit_transform(synth_seq).toarray()
            
            # Compute the pairwise DTW distances
            dtw_distances = np.array([
                dtw(real_encoded[i], synth_encoded[j])
                for i in range(len(real_encoded))
                for j in range(len(synth_encoded))
            ])
            
            # Compute the MMD score using the formula
            # MMD^2 = (1/n^2) * sum(dtw(real_i, real_j)) + (1/m^2) * sum(dtw(synth_i, synth_j)) - (2/nm) * sum(dtw(real_i, synth_j))
            n, m = len(real_encoded), len(synth_encoded)
            real_real_sum = dtw_distances[:n*n].sum()
            synth_synth_sum = dtw_distances[n*n:].sum()
            real_synth_sum = dtw_distances[n*n:(n+m)*(n+m)].sum()
            mmd_score = (real_real_sum / (n*n)) + (synth_synth_sum / (m*m)) - (2 * real_synth_sum / (n*m))
            
            # Add the MMD score to the total
            total_mmd += mmd_score
            num_pairs += 1
    
    # Compute the average MMD score
    avg_mmd = total_mmd / num_pairs
    return avg_mmd

def main():
    # Load data frames from CSV files
    real_data = pd.read_csv('data/dvlog/preprocessed/acoustic.encoded.csv')
    synth_data = pd.read_csv('sampled/sampled_dvlog_cleanedAndSorted.csv')
    index_column = "Timestamp"
    target_column = "Label"
    
    # Call the mmd_dtw function
    #avg_mmd = mmd_dtw(real_data, synth_data, index_column)
    #print(f'Average MMD Score: {avg_mmd}')
    desired_seq_length =1000

    accuracy = sequence_classifier(real_data, synth_data, index_column, target_column, desired_seq_length)
    print("Prediction accuracy on synthetic data:", accuracy)

    accuracy = sequence_classifier(real_data, real_data, index_column, target_column, desired_seq_length)
    print("Prediction accuracy on real data:", accuracy)

if __name__ == "__main__":
    main()
