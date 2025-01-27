import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam

# Step 1: Load and Normalize the Dataset
file_features = pd.read_csv('engineered_features.csv')

# Normalize continuous features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(file_features[['total_file_access', 'to_removable_media', 
                                                          'from_removable_media', 'hour_mean', 'hour_std']])

file_features[['total_file_access', 'to_removable_media', 
               'from_removable_media', 'hour_mean', 'hour_std']] = normalized_features

# Step 2: Calculate Per-User Baseline
user_stats = file_features.groupby('user').agg({
    'total_file_access': ['mean', 'std'],
    'to_removable_media': ['mean', 'std'],
    'from_removable_media': ['mean', 'std'],
    'hour_mean': ['mean', 'std'],
    'hour_std': ['mean', 'std']
})

user_stats.columns = ['_'.join(col) for col in user_stats.columns]
user_stats = user_stats.reset_index()

# Merge back to the original dataset
file_features = file_features.merge(user_stats, on='user', how='left')

# Step 3: Flag Anomalies Based on Deviation from Baseline
def is_anomalous(row, features):
    for feature in features:
        if abs(row[feature] - row[f'{feature}_mean']) > 3 * row[f'{feature}_std']:
            return 1
    return 0

features_to_check = ['total_file_access', 'to_removable_media', 'from_removable_media', 'hour_mean', 'hour_std']
file_features['anomaly'] = file_features.apply(is_anomalous, axis=1, features=features_to_check)

# Save baseline-labeled data
file_features.to_csv('baseline_labeled_features.csv', index=False)

# Step 4: Synthetic Anomaly Generation using GAN
# Prepare data for GAN
normal_data = file_features[file_features['anomaly'] == 0][features_to_check].values

# Generator Model
def build_generator(input_dim, output_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(output_dim, activation='tanh')
    ])
    return model

# Discriminator Model
def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

# Set GAN parameters
input_dim = normal_data.shape[1]
generator = build_generator(input_dim, input_dim)
discriminator = build_discriminator(input_dim)
gan = build_gan(generator, discriminator)

# Compile models
discriminator = Sequential([
    Dense(64, input_dim=5),
    LeakyReLU(alpha=0.2),
    Dense(32),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')
])

discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Create dummy data
dummy_real_samples = np.random.normal(0, 1, (64, 5))  # Shape matches input_dim
dummy_real_labels = np.ones((64, 1))  # Labels for real samples

# Train discriminator on dummy data
d_loss_real = discriminator.train_on_batch(dummy_real_samples, dummy_real_labels)
print(f"Discriminator loss on dummy data: {d_loss_real}")

# Train GAN
# Training loop
epochs = 5000
batch_size = 64
for epoch in range(epochs):
    # Select random batch from normal data
    idx = np.random.randint(0, normal_data.shape[0], batch_size)
    real_samples = normal_data[idx]
    print("Expected input shape:", discriminator.input_shape)
    print("Real samples shape:", real_samples.shape)

    # Generate synthetic samples
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    fake_samples = generator.predict(noise)
    
    # Ensure data dimensions match
    assert real_samples.shape == fake_samples.shape, "Mismatch in real and fake sample shapes"
    
    # Train discriminator
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train generator
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")


# Generate Synthetic Anomalies
synthetic_anomalies = generator.predict(np.random.normal(0, 1, (1000, input_dim)))

# Save synthetic anomalies
synthetic_anomalies = scaler.inverse_transform(synthetic_anomalies)  # Inverse transform to original scale
synthetic_anomalies_df = pd.DataFrame(synthetic_anomalies, columns=features_to_check)
synthetic_anomalies_df.to_csv('synthetic_anomalies.csv', index=False)

print("Baseline anomalies and synthetic anomalies generated successfully.")
