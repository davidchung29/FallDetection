# Fall Detection via Transfer Learning with ResNet-50

**Author**: David Chung 
**Project Date** November 2022 - January 2023
**Collaborators**: Under the mentorship of Prof. Amit Roy-Chowdhury and PhD student Abhishek Aich, UC Riverside  
**Technologies**: PyTorch · OpenCV · ResNet-50 · Transfer Learning · Lambda Labs GPU  

---

## 🧠 Overview

This project explores the application of deep learning for real-time fall detection using convolutional neural networks (CNNs). Falls are a major public health concern, and this system is designed to detect them accurately from live video input. We developed a two-stage pipeline using **transfer learning with ResNet-50**, first training on the **UCF101 action recognition dataset** and then fine-tuning on the **E-FPDS fall detection dataset**. The final model achieved **99.5% accuracy** in fall classification — outperforming state-of-the-art methods such as YOLO-based TD architectures.

---

## 🧪 Project Highlights

- ✅ **Two-phase deep learning pipeline**
  - Stage 1: ResNet-50 trained on UCF101 (101-class action recognition).
  - Stage 2: ResNet-50 fine-tuned for binary classification (fall vs. non-fall) using E-FPDS.

- 📁 **Custom preprocessing pipeline**
  - Extracted video frames with OpenCV.
  - Augmentation: center cropping, horizontal flipping, normalization.
  - Dataset balancing & stratified train/val splits.

- 📊 **Performance**
  - **UCF101** (pretraining): 99.8% validation accuracy.
  - **E-FPDS** (fine-tuning): **99.5% fall detection accuracy**, outperforming TD-model (98.97%).

- 🚀 **Accelerated training**
  - All models trained using Lambda Labs eGPU instance with GPU acceleration for reproducibility and efficiency.

---

## 📂 Dataset Info

| Dataset   | Description                                   | Usage              |
|-----------|-----------------------------------------------|--------------------|
| UCF101    | 13K+ video clips of 101 human actions          | ResNet pretraining |
| E-FPDS    | Annotated still frames of falls and non-falls | Fine-tuning        |

- UCF101 source: [UCF CRCV](https://www.crcv.ucf.edu/data/UCF101.php)  
- E-FPDS source: [E-FPDS Download](https://gram.web.uah.es/data/datasets/fpds/index.html)

---

## 🏗️ Architecture

```text
[Raw Videos] 
   ↓
[OpenCV Frame Extraction] 
   ↓
[Frame Preprocessing & Augmentation]
   ↓
[ResNet-50 (UCF101 Pretraining)]
   ↓
[ResNet-50 (E-FPDS Fine-tuning)]
   ↓
[Binary Classification Output: FALL / NO FALL]
```

- Loss Function: CrossEntropyLoss  
- Optimizer: SGD (lr=0.001, momentum=0.9)  
- Learning Rate Scheduler: StepLR (step_size=7, gamma=0.1)  
- Epochs: 5 (UCF101) + 5 (E-FPDS)

---

## 🔍 Results

| Metric                   | UCF101          | E-FPDS         |
|--------------------------|------------------|----------------|
| Training Accuracy        | 99.8%           | 98.4%          |
| Validation Accuracy      | 100%            | **99.5%**      |
| Avg Training Loss        | 0.00806         | 0.00           |
| Avg Validation Loss      | 0.00018         | 0.00           |

> 📈 E-FPDS model surpassed the performance of prior SOTA models including YOLOv6 and TD-model benchmarks.

---

## ⚙️ Setup Instructions

1. **Environment**
   ```bash
   pip install torch torchvision torchaudio opencv-python scikit-learn matplotlib
   ```

2. **Preprocess UCF101 and E-FPDS**
   - Use OpenCV to extract frames into action/fall folders.
   - Resize to `200x200` (UCF101) and `480x640` (E-FPDS).

3. **Train the Model**
   ```python
   # Step 1: Train on UCF101
   python train_ucf101.py

   # Step 2: Fine-tune on E-FPDS
   python train_efpds.py
   ```

---

## 📈 Future Work

- Expand to multi-person fall detection using bounding boxes
- Deploy as a web or mobile service for healthcare applications

---

## 🙏 Acknowledgments

This project was conducted under the guidance of **Prof. Amit Roy-Chowdhury** and **PhD student Abhishek Aich** at **UC Riverside** as part of a research initiative to explore AI in public health applications.

---
