import pandas as pd

# Step 1: Load and Filter `insiders.csv` for Dataset 6.1
# Load insiders.csv
insiders_df = pd.read_csv('insiders.csv')

# Convert 'start' and 'end' columns to datetime
insiders_df['start'] = pd.to_datetime(insiders_df['start'])
insiders_df['end'] = pd.to_datetime(insiders_df['end'])

# Filter for dataset 6.1
insiders_6_1 = insiders_df[insiders_df['dataset'] == 6.1]

# Step 2: Load `file.csv` and Merge with `insiders_6_1`
# Load file.csv
file_df = pd.read_csv('file.csv')

# Convert 'date' column in file.csv to datetime
file_df['timestamp'] = pd.to_datetime(file_df['date'])

# Merge file.csv with insiders_6_1 based on user
file_df = pd.merge(file_df, insiders_6_1, left_on='user', right_on='user', how='left')

# Label file operations as anomalous if they fall within the insider time range
file_df['label'] = file_df.apply(
    lambda row: 1 if pd.notna(row['start']) and row['start'] <= row['timestamp'] <= row['end'] else 0, axis=1
)

# Step 3: Feature Engineering
## Temporal Features
file_df['hour'] = file_df['timestamp'].dt.hour
file_df['is_after_hours'] = file_df['hour'].apply(lambda x: 1 if x < 6 or x > 18 else 0)

## Behavioral Features
file_df['removable_media_usage'] = file_df['to_removable_media'] + file_df['from_removable_media']

## Scenario-Specific Features
# Sensitive file access based on keywords
# file_df['sensitive_file_access'] = file_df['file_tree'].str.contains('confidential|password', case=False, na=False).astype(int)
file_df['sensitive_file_access'] = 0  # Placeholder feature

# File deletions as a feature
file_df['is_deletion'] = (file_df['activity'] == 'delete').astype(int)

# Combine Features
file_df['anomaly_indicator'] = (
    file_df['is_after_hours'] |
    (file_df['removable_media_usage'] > 0) |
    file_df['sensitive_file_access'] |
    file_df['is_deletion']
).astype(int)

# Step 4: Normalize Continuous Features
from sklearn.preprocessing import StandardScaler

# Select continuous features to normalize
continuous_features = ['removable_media_usage', 'hour']
scaler = StandardScaler()
file_df[continuous_features] = scaler.fit_transform(file_df[continuous_features])

# Step 5: Save the Labeled and Engineered Dataset
file_df.to_csv('features_6_1.csv', index=False)

# Display a summary of the final dataset
print("Labeled Dataset Saved: features_6_1.csv")
print(file_df[['label', 'anomaly_indicator']].value_counts())
