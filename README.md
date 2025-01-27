# File Integrity Monitoring Enhancement with CNN and GAN

## Overview
This repository contains a project focused on enhancing File Integrity Monitoring (FIM) systems using deep learning techniques. By integrating Convolutional Neural Networks (CNNs) for anomaly detection and Generative Adversarial Networks (GANs) for synthetic data generation, the project aims to improve the accuracy and effectiveness of detecting unauthorized file operations.

### Goals
- Address the limitations of traditional FIM systems, including high false positives and difficulty detecting sophisticated threats like insider attacks and fileless malware.
- Leverage CNNs to identify nuanced patterns in file activity logs.
- Generate synthetic anomalies using GANs to enrich datasets and address class imbalances.

---

## Project Highlights

### Motivation
FIM is a critical security control mandated by regulatory standards such as PCI-DSS, HIPAA, and GDPR. Traditional systems often lack the contextual awareness needed to identify complex threats. This project integrates advanced deep learning methods to:
- Reduce false positives.
- Improve detection accuracy.
- Adapt to emerging threats in large-scale systems.

### Components
1. **Dataset Preparation** (`project.py`):
   - Merges insider activity logs with file access data.
   - Labels file operations based on insider threat time ranges.
   - Engineers features like after-hours access, removable media usage, and sensitive file interactions.

2. **Anomaly Detection with CNN** (`CNN.py`):
   - Trains a CNN model to classify file operations as normal or anomalous.
   - Uses structured file activity logs as input.
   - Evaluates the model and provides metrics like accuracy, confusion matrix, and classification reports.

3. **Synthetic Data Generation with GAN** (`GAN.py`):
   - Generates synthetic anomalies to address dataset limitations.
   - Enhances diversity in training data for the CNN model.
   - Saves generated anomalies for further analysis.

---

## How It Works

### CNN Model
- **Architecture:** Three convolutional layers with ReLU activations, max pooling, and dropout layers to prevent overfitting.
- **Output:** Binary classification (normal vs. anomalous).
- **Loss Function:** Binary cross-entropy.
- **Optimizer:** Adam.

### GAN Model
- **Generator:** Produces synthetic anomalies based on patterns observed in existing data.
- **Discriminator:** Distinguishes between real and synthetic anomalies.
- **Training:** Iterative adversarial process to enhance the quality of generated data.

---

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: TensorFlow, Keras, pandas, scikit-learn, Matplotlib
- Ensure access to the following files:
  - `insiders.csv`: Insider threat logs.
  - `file.csv`: File activity logs.

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/ozaksen/FileOps-Anomaly-Detection.git
cd FileOps-Anomaly-Detection
pip install -r requirements.txt
```

### Usage
1. **Prepare the Dataset**:
   ```bash
   python project.py
   ```
2. **Train the CNN Model**:
   ```bash
   python CNN.py
   ```
3. **Generate Synthetic Anomalies**:
   ```bash
   python GAN.py
   ```

---

## Results
- **Improved Detection:** Enhanced anomaly detection accuracy through CNNs.
- **Dataset Augmentation:** GAN-generated anomalies enriched the training data, addressing imbalances and improving recall.

### Future Enhancements
- Real-time anomaly detection.
- Broader feature engineering for diverse threat scenarios.
- Integration with live FIM systems for enterprise-scale deployment.

---

## References
- CERT Insider Threat Dataset
- Regulatory Standards: PCI-DSS, HIPAA, GDPR
- Relevant literature on anomaly detection and deep learning.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

Contributions and feedback are welcome! Please submit issues or pull requests to help improve this repository.

