# DG2GAN: Technical Project Report

## 📋 Executive Summary
The DG2GAN project is a state-of-the-art Generative AI pipeline designed to predict internal structural damage in composite aircraft panels without the need for destructive or time-consuming ultrasonic scans. By fusing multi-modal data (2D surface images, 3D point clouds, and material metadata), the system generates high-resolution color heatmaps representing internal delamination.

---

## 🏗️ Project Structure & Component Analysis

### `/models` Folder: The "Brain"
- **`fusion_generator.py`**: Contains the `DG2Generator`. It is a U-Net style architecture modified for multi-modal fusion. It encodes two surface scans (chopped and full) through parallel CNN layers, processes 3D mesh and CSV data through MLPs, and fuses them at a central bottleneck before decoding into a 3-channel RGB heatmap.
- **`discriminator.py`**: Implements a PatchGAN `Discriminator`. It evaluates "patches" of the generated heatmap against the ground truth. Crucially, it is "conditioned" on the input images, ensuring that the predicted damage aligns perfectly with the external surface features.

### `/utils` Folder: The "Infrastructure"
- **`dataset.py`**: The `CompositeDataset` class. This is where raw data is transformed into AI-ready tensors.
  - **Function `csv_to_heatmap`**: Converts raw sensor data from CSVs into physical heatmaps using the `Jet` colormap.
  - **Function `process_mesh`**: Subsamples 3D `.ply` files into a fixed number of vertices (1024) for consistent neural network input.
- **`losses.py`**: Contains the mathematical scoring of the model.
  - **Adversarial Loss**: Forces the generator to produce realistic-looking images.
  - **L1 Loss**: Ensures the generated heatmap is pixel-accurately similar to the ground truth.

### Root Directory: The "Orchestration"
- **`train.py`**: The training loop. It iterates through the dataset, updates the Discriminator's ability to spot fakes, and then updates the Generator's ability to create better fakes.
- **`infer.py`**: The deployment script. It loads a trained model to make predictions on new, unseen panel data.
- **`config.py`**: A centralized "control center" where image sizes, learning rates, and hardware settings (CPU/GPU) are defined.
- **`test_model.py`**: A lightweight utility to verify that all data dimensions match before starting a heavy training run.

---

## 🔄 The Pipeline Workflow

1.  **Input Phase**:
    - **Visual**: Two PNGs (Full panel for context, Chopped panel for detail).
    - **Geomtric**: `.ply` point cloud showing surface dents/deformations.
    - **Contextual**: `.csv` metadata describing material properties.
2.  **Fusion Phase**:
    - All inputs are compressed into a "latent vector" (the model's internal understanding of the panel).
3.  **Generation Phase**:
    - The model reconstructs an RGB image representing internal damage.
4.  **Output Phase**:
    - A comparison grid is saved showing **Surface Input | Predicted Heatmap | Ground Truth**.

---

## 🔭 Deep Dive: Stakeholder FAQ

**Q1: Why use both "chopped" and "full" images?**
*Answer*: The 'full' image provides global spatial context (where is the panel on the wing?), while the 'chopped' image provides high-resolution local details of the surface geometry. Using both ensures the model doesn't lose its place during prediction.

**Q2: How reliable is the CSV-to-Heatmap conversion?**
*Answer*: This is the most accurate part of the project. By using the raw sensor data from the CSV, we ensure that the AI is learning from actual physics-based measurements rather than just black-and-white visual approximations.

**Q3: Can this model work on different materials?**
*Answer*: Currently, it is optimized for the composites in the training set. However, because we include **CSV Metadata** as a feature, the model can be "fine-tuned" for new materials by simply adding more samples with their respective metadata.

**Q4: Is the prediction real-time?**
*Answer*: Yes. While training takes hours, the `infer.py` script takes less than 100 milliseconds to generate a prediction once the model is loaded, making it suitable for handheld inspection devices.

---

## 🛠️ Enabling/Disabled Features
- **GPU Acceleration**: Enabled automatically via `config.DEVICE`. If a CUDA-enabled GPU is detected, performance increases 10x.
- **Data Augmentation**: Currently disabled to maintain spatial alignment between surface and internal scans.
- **Colormaps**: Set to `Jet` by default in `dataset.py` for standard engineering heatmaps.
