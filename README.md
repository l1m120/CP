# 🚦 Development of a Standalone and Open-Source Vehicle Detection Framework Using YOLO for Traffic Analysis

> **A Privacy‑Compliant, Open‑Source Vehicle Detection Framework for Malaysian Traffic Analysis**

![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)
![Framework](https://img.shields.io/badge/Framework-YOLO11%20%7C%20Streamlit-red)
![Dataset](https://img.shields.io/badge/Dataset-Zenodo-blueviolet)
![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16866508-blue)
![Open Access](https://img.shields.io/badge/Open%20Access-Yes-brightgreen)
![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-ff4b4b)
![GUI](https://img.shields.io/badge/GUI-Interactive-informational)

---

## 📖 Overview

This project entails a complete, end‑to‑end computer vision framework designed to address the lack of localized, privacy‑compliant traffic datasets in Southeast Asia—particularly Malaysia. The project delivers:

* **MY‑VID**: an open‑source Malaysian vehicle image dataset
* **A fully reproducible training pipeline** (Jupyter Notebook)
* **An optimized YOLO11 model** for vehicle detection
* **A deployed Streamlit GUI** for real‑time analysis

The framework aligns with **JKR Malaysia vehicle taxonomy and data standards**, ensuring real‑world relevance and deployability.

---

## 📂 Project Structure & Research Outcomes (RO)

This repository is organized around four core **Research Objectives (ROs)**:

| Research Objective | Description       | Deliverable / Location                                                                                         |
| ------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------- |
| **RO1**            | Data Acquisition  | **MY‑VID Dataset (Zenodo)**   [https://doi.org/10.5281/zenodo.16866508](https://doi.org/10.5281/zenodo.16866508) *(External – 8,832 images)*|
| **RO2**            | Training Pipeline | `MY-VID_End_to_End_Pipeline.ipynb`                                                                             |
| **RO3**            | Optimal Model     | `models/best.pt` *(YOLO11s fine‑tuned weights)*                                                                |
| **RO4**            | Deployment (GUI)  | Live App: [https://trafficsense-ai.streamlit.app/](https://trafficsense-ai.streamlit.app/)  Source: `outputgui.py`                  |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/l1m120/CP.git
cd CP
```

### 2️⃣ Install Dependencies

Ensure **Python 3.10+** is installed.

```bash
pip install -r requirements.txt
```

> **Note**: For GPU acceleration, install a PyTorch build compatible with your local **CUDA** version.

---

## 🛠️ Component Details

### 🔹 RO1: MY‑VID Dataset

**MY‑VID (Malaysian Vehicle Image Dataset)** is the first open‑source, privacy‑compliant vehicle dataset specifically curated for Malaysian road environments.

**Key Features**:

* **Classes (JKR‑standard, 6)**: Car, Van, Light Lorry, Heavy Lorry, Bus, Motorcycle
* **Source**: Roadside imagery on Malaysian highway
* **Privacy**: Non‑road regions automatically blurred
* **Hosting**: Zenodo (external, due to large size)

🔗 **Dataset Access**: [https://doi.org/10.5281/zenodo.16866508](https://doi.org/10.5281/zenodo.16866508)

---

### 🔹 RO2: Training Pipeline

The notebook **`MY-VID_End_to_End_Pipeline.ipynb`** provides a fully reproducible workflow:

**Pipeline Stages**:

1. Video‑to‑frame extraction
2. Automated privacy masking
3. Annotation processing (LabelMe JSON → YOLO TXT)
4. YOLO11 model training and evaluation

**How to Run**:

```bash
jupyter notebook MY-VID_End_to_End_Pipeline.ipynb
```

This ensures transparency and reproducibility for academic and industrial users.

---

### 🔹 RO3: Optimal Model Weights

The best‑performing model from the study is provided for direct reuse.

* **Model**: YOLO11s
* **Path**: `models/best.pt`
* **Performance**:

  * mAP@0.5: **0.774**
  * Inference Speed: **~666 FPS** (GPU)

**Quick Usage Example**:

```python
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO("models/best.pt")

# Run inference
results = model("path/to/your/image.jpg")
```

---

### 🔹 RO4: TrafficSense AI (GUI)

TrafficSense AI includes an interactive **Streamlit‑based GUI** for practical deployment.

**Features**:

* Image upload
* Real‑time vehicle detection
* Class‑wise analytics and visualization

🔗 **Live Demo**: [https://trafficsense-ai.streamlit.app/](https://trafficsense-ai.streamlit.app/)

**Run Locally**:

```bash
streamlit run outputgui.py
```

---

## 📜 Citation

If you use **MY‑VID**, the training pipeline, or TrafficSense AI in your research, please cite:

```bibtex
@dataset{myvid_v2_2025,
  author       = {Lim Zi Xuan},
  title        = {MY-VID v2: Malaysian Vehicle Image Dataset (JKR Taxonomy)},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v2.0},
  doi          = {10.5281/zenodo.16866508}
}
```

---

## 📄 License

* **Code**: MIT License (see `LICENSE`)
* **Dataset (MY‑VID)**: Creative Commons **CC BY 4.0**

---

<p align="center">
  <b> &copyDeveloped by Lim Zi Xuan · Sunway University · 2025</b>
</p>
