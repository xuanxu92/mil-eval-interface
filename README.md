# MIL Interface for Evaluating Pathology Foundation Models

This repository provides a Multiple Instance Learning (MIL) based interactive interface using Gradio to evaluate different pathology foundation models on downstream tasks such as cancer subtype classification.

## 🔧 Features

- MIL-based inference pipeline for pathology features
- Interactive interface built with Gradio
- Flexible support for features extracted from various foundation models
- Easily configurable with CSV label files and extracted features

---

## 🛠️ Installation

Please install the environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate mil-eval
```

*Note: If your environment name differs, adjust accordingly.*

---

## 📁 Data Preparation

### 1. Extract Patch-level Features

Use your pathology foundation model (e.g., CLIP, PathDINO, etc.) to extract patch-level features from WSIs. Organize them into a single folder.

**Example folder structure:**

```
features/
├── sample_1.pt
├── sample_2.pt
└── ...
```

### 2. Prepare CSV for Labels

Create a CSV file containing the mapping between each sample's feature file and its corresponding ground truth label.

**Example CSV:**

```csv
case_id,label
sample_1,LUAD
sample_2,LUSC
...
```

---

## ✏️ Configuration

Before running the interface, update the following parameters in `app_example_final.py`:

- **Path to feature folder:**  
  Replace the placeholder path in the code:
  ```python
  pancancer_path = "/path/to/your/features/"
  ```

- **Path to CSV label file:**  
  Modify the corresponding DataFrame loading line to point to your CSV:
  ```python
  df = pd.read_csv("/path/to/your/labels.csv")
  ```

---

## 🚀 Run the App

Once configured, launch the Gradio interface:

```bash
python app_example_final.py
```

The app will start locally and open in your browser, allowing interactive evaluation and visualization of different pathology foundation models.

---

## 📌 Notes

- This pipeline supports evaluating patch-level features with ground truth labels via MIL aggregation.
- You can extend this interface to support new tasks or integrate additional models as needed.

---

## 📬 Contact

For questions or collaborations, please reach out via GitHub issues.

