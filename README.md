# File Anomaly Detection with CNN and GAN

This repository contains the implementation of a deep learning-based anomaly detection system for file access operations. It leverages a **Convolutional Neural Network (CNN)** for anomaly classification and a **Generative Adversarial Network (GAN)** for generating synthetic anomalous data. Additionally, it includes data preprocessing and feature engineering steps to create a robust dataset for the models.

---

## Overview

This project is designed to detect anomalous file access patterns in system logs, using the following components:

- **CNN Model** (`CNN.py`): A binary classification model to identify anomalous file access operations.
- **GAN Model** (`GAN.py`): Generates synthetic anomalies for augmenting training datasets.
- **Dataset Preparation** (`project.py`): Prepares labeled datasets by merging log files with insider threat data and applying feature engineering.

---

## File Descriptions

### 1. **`project.py`**
   - **Purpose**: Prepares the labeled dataset for anomaly detection.
   - **Features**:
     - Merges user activity logs with insider threat data.
     - Labels file operations as normal or anomalous.
     - Engineers temporal and behavioral features.
     - Normalizes continuous features.
   - **Output**: Generates a labeled dataset (`features_6_1.csv`).

### 2. **`CNN.py`**
   - **Purpose**: Trains a CNN to classify file operations as normal or anomalous.
   - **Steps**:
     1. Loads the preprocessed dataset (`features_6_1.csv`).
     2. Normalizes features and splits the data into training and testing sets.
     3. Defines and trains a CNN architecture.
     4. Evaluates the model and visualizes performance metrics.
     5. Saves the trained model as `cnn_file_operations_6_1.h5`.

### 3. **`GAN.py`**
   - **Purpose**: Generates synthetic anomalies to enhance the dataset.
   - **Steps**:
     1. Loads normalized features.
     2. Flags baseline anomalies based on statistical deviations.
     3. Defines GAN architecture with a generator and discriminator.
     4. Trains the GAN to generate synthetic anomalous samples.
     5. Saves synthetic anomalies as `synthetic_anomalies.csv`.

### 4. **`cnn_file_access_model.h5`**
   - **Purpose**: Pre-trained CNN model for anomaly classification.
   - **Usage**: Can be directly loaded and used for predictions.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/file-anomaly-detection.git
   cd file-anomaly-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the necessary CSV files (`insiders.csv`, `file.csv`, etc.) are available in the working directory.

---

## Usage

1. **Dataset Preparation**:
   - Run `project.py` to generate the labeled dataset.
     ```bash
     python project.py
     ```

2. **Train the CNN Model**:
   - Run `CNN.py` to train the anomaly classification model.
     ```bash
     python CNN.py
     ```

3. **Generate Synthetic Anomalies**:
   - Run `GAN.py` to create additional anomalies.
     ```bash
     python GAN.py
     ```

---

## Key Results

- **Classification Metrics**: The CNN model achieves robust accuracy in detecting anomalous file operations.
- **Synthetic Anomalies**: GAN-generated data improves training diversity and enhances model generalization.

---

## Future Enhancements

- Expand feature engineering for more complex scenarios.
- Implement real-time anomaly detection in file systems.
- Explore unsupervised learning for anomaly detection.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to enhance the functionality of this project.

---

## License

This project is licensed under the [MIT License](LICENSE).



