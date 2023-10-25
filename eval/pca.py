import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Sample data frames
real_data = pd.read_csv('data/dvlog/preprocessed/acoustic.encoded.csv')
synth_data = pd.read_csv('sampled/sampled_dvlog_cleanedAndSorted.csv')

# Combine the data frames
combined_data = pd.concat([real_data, synth_data], axis=0)

# Separate out the features and labels
# Exclude the 'Label' column for PCA
features = combined_data.columns.difference(['Label'])
x = combined_data[features].values
y = combined_data['Label'].values

# Standardize the features
x = StandardScaler().fit_transform(x)


# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
principal_df['Label'] = y


# Split the PCA results for real_data and synth_data
real_principal_df = principal_df.iloc[:len(real_data)]
synth_principal_df = principal_df.iloc[len(real_data):]


# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(real_principal_df['Principal Component 1'], real_principal_df['Principal Component 2'], c=real_principal_df['Label'], cmap='rainbow', alpha=0.5)
plt.title('PCA Projection of Real Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.savefig('real.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(synth_principal_df['Principal Component 1'], synth_principal_df['Principal Component 2'], c=synth_principal_df['Label'], cmap='rainbow', alpha=0.5)
plt.title('PCA Projection of Synthetic Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.savefig('synth.png')
plt.show()

