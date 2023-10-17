import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tslearn.metrics import dtw
import numpy as np

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
    for real_seq in real_sequences:
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
    real_data = pd.read_csv('real_data.csv')
    synth_data = pd.read_csv('synth_data.csv')
    
    # Call the mmd_dtw function
    avg_mmd = mmd_dtw(real_data, synth_data, 'sequence_index')
    print(f'Average MMD Score: {avg_mmd}')

if __name__ == "__main__":
    main()
