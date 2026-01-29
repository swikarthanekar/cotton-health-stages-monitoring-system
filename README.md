# Agri-Vision: Cotton Crop Maturity & Health Classifier

## Overview

This project implements a **computer vision system for agricultural decision support** that analyzes cotton crop images to determine:

- **Growth Phase** (Vegetative → Harvest Ready)  
- **Visual Health Status** (damaged or healthy)  
- **Health Score (0–100)**  
- **Explainability via Grad-CAM heatmaps**  

Unlike basic crop classifiers, this system combines **transfer learning, visual reasoning, explainable AI, and real-time deployment** to simulate how modern agri-vision tools help farmers decide the correct harvest time.

---

## Problem Statement

Cotton maturity is traditionally judged through manual inspection, which is:

- subjective  
- inconsistent  
- time-consuming  

This project builds an automated pipeline that can:

1. Recognize the **growth stage** of cotton.
2. Detect **visual damage cues**.
3. Assign a **health score**.
4. Provide **explainability** using Grad-CAM.
5. Return structured **JSON output** through an API.

---

## Growth Phases

| Phase | Description |
|-------|------------|
| Phase 1 | Vegetative / Budding |
| Phase 2 | Flowering |
| Phase 3 | Bursting (Partially Opened Boll) |
| Phase 4 | Harvest Ready (Fully Open Cotton) |

---

## Datasets Used

This project combines **three public datasets**:

### 1. Cotton Boll Dataset (905 images)  
Annotated in Pascal VOC format  
Used for **Phase 3 & Phase 4**

🔗 https://www.kaggle.com/datasets/kanishbkhagat/cotton-boll-dataset

---

### 2. Cotton Leaf Disease Dataset  
Classes:
- bacterial_blight  
- curl_virus  
- fusarium_wilt  
- healthy  

Used to **inform health scoring logic**

🔗 https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset

---

### 3. Cotton Boll & Flowers Recognition Dataset  
Mixed growth stage cotton images

🔗 https://www.kaggle.com/datasets/sweefongwong/cotton-boll-and-flowers-recognition-dataset

---

## Dataset Organization

```
data/
├── train/
│   ├── phase1_vegetative/
│   ├── phase2_flowering/
│   ├── phase3_bursting/
│   └── phase4_harvest/
│
└── val/
    ├── phase1_vegetative/
    ├── phase2_flowering/
    ├── phase3_bursting/
    └── phase4_harvest/
```

---

## Model Architecture

- **Backbone:** ResNet-18 (pretrained on ImageNet)  
- **Strategy:** Transfer Learning  
- **Loss:** CrossEntropy  
- **Optimizer:** Adam  
- **Input Size:** 224×224  

The final fully-connected layer is replaced with a **4-class classifier head**.

---

## Training

From the `model/` folder:

```bash
python train.py
```

Outputs:
- `cotton_stage_model.pth`
- `training_curve.png`

Final validation accuracy ≈ **75%**

---

## Explainability (Grad-CAM)

Visualizes where the CNN focuses while predicting.

```bash
cd explainability
python gradcam.py ../data/train/phase3_bursting/3.jpg
```

Output:
- `gradcam_output.png`

Red regions indicate the most influential areas.

---

## Inference API

The trained model is deployed as a **Flask REST API**.

### Start server:
```bash
cd inference
python api.py
```

### Send test request:
```bash
python test_request.py
```

### Example Output

```json
{
  "stage": "Phase 3 - Bursting",
  "is_ripped": true,
  "health_score": 76
}
```

---

## Health Score Logic

The health score is derived from:

- model confidence  
- visual damage cues  
- cotton maturity stage  

The leaf disease dataset is used as a **visual reference distribution**, not as a supervised classifier.

This mimics early agri-decision support systems.

---

## Project Structure

```
cotton-vision/
├── data/
├── model/
├── explainability/
├── inference/
├── demo_images/
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python  
- PyTorch  
- OpenCV  
- Flask  
- torchvision  
- Grad-CAM  

---

## Final Outcome

This system demonstrates:

- Multi-stage cotton maturity classification  
- Visual explainability  
- Health estimation  
- Real-time API inference  

It forms a **foundation for AI-powered agricultural decision support systems**.
