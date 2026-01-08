# Improving Machine Learning-based Early Diagnosis <br> Framework of Degenerative Brain Disease through <br> Self-Supervised Learning

> **KIISE Korea Computer Congress 2025 (KCC 2025)**, pp. 2105–2107  

This repository contains the official implementation of the paper **"Improving Machine Learning-based Early Diagnosis Framework of Degenerative Brain Disease through Self-Supervised Learning"**.

We propose a framework that utilizes **Self-Supervised Learning (SSL)** based on SimCLR to learn multimodal feature representations (Genotype SNP + Phenotype MRI + Demographics). This approach aims to improve the early diagnosis performance (AUC/mAUC) of Alzheimer's Disease (AD) and Mild Cognitive Impairment (MCI), addressing the issue of label scarcity in medical imaging.

---

## 📝 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Tasks & Metrics](#tasks--metrics)
- [Results](#results)
- [Environment](#environment)
- [How to Run](#how-to-run)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## 🔍 Overview

Alzheimer's Disease (AD) is a representative neurodegenerative disorder. Accurate diagnosis during the **Mild Cognitive Impairment (MCI)** stage, a prodromal phase of AD, is critical for effective disease management. However, acquiring high-quality labeled data in medical imaging research is challenging.

To mitigate the label scarcity problem, this study introduces an **SSL-based representation learning** framework.

**Key Contributions:**
1.  **Multimodal Integration**: We construct input vectors by combining ROI-based MRI features, SNPs, and demographic information.
2.  **Self-Supervised Learning**: We pre-train feature representations (embeddings) without labels using a **SimCLR-based contrastive learning** approach.
3.  **Performance Improvement**: The pre-trained encoder embeddings are fed into downstream classifiers (SVC, GPC, RFC), demonstrating superior performance compared to supervised baselines.

---

## 📊 Dataset

We utilized the **ADNI-1** (Alzheimer's Disease Neuroimaging Initiative) dataset.

* **Total Subjects**: 734
    * **CN** (Cognitive Normal): 211
    * **MCI** (Mild Cognitive Impairment): 350
    * **AD** (Alzheimer's Disease): 173
* **Input Features**:
    * **Demographics**: Sex (Binary), Education Length, Age.
    * **MRI**: 93 ROI-based Gray Matter Volume features (after preprocessing).
    * **SNP**: 2,098 AD-associated SNPs.
    * **Final Dimension**: **2,191-dim** (Multimodal concatenated).

### Definition of sMCI / pMCI
MCI subjects were categorized based on a **30-month** follow-up:
* **pMCI (Progressive MCI)**: Converted to AD within 30 months.
* **sMCI (Stable MCI)**: Did not convert to AD within 30 months.

---

## 🛠 Methods

### Baseline Classifiers (scikit-learn)
* **SVC**: Linear kernel, `C=1.0`
* **GPC**: Gaussian kernel (RBF)
* **RFC**: Random Forest, `n_estimators=100`

### SSL Representation Learning (SimCLR, PyTorch)
* **Loss Function**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
* **Embedding Dimension**: 128
* **Pre-training Epochs**: 20
* **Workflow**: After pre-training, the projection head is removed, and the encoder's output embedding is used as input for the downstream classifiers.

---

## 🎯 Tasks & Metrics

### Classification Tasks
1.  **Binary Classification**:
    * CN vs. AD
    * CN vs. MCI
    * sMCI vs. pMCI
2.  **Multi-class Classification**:
    * 3-Class: CN vs. MCI vs. AD
    * 4-Class: CN vs. sMCI vs. pMCI vs. AD

### Evaluation
* **Metrics**: AUC (Binary) / mAUC (Multi-class)
* **Validation**: 5-fold Cross-Validation

---

## 📈 Results

Performance comparison using 5-fold CV (Mean ± Std).

| Model | Metric | CN/AD | CN/MCI | sMCI/pMCI | CN/MCI/AD | CN/sMCI/pMCI/AD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVC** | (m)AUC | 0.8811 ± 0.0281 | 0.6947 ± 0.0449 | 0.6683 ± 0.0201 | 0.7004 ± 0.0187 | 0.7111 ± 0.0084 |
| **SSL-SVC** | (m)AUC | 0.9271 ± 0.0207 | 0.7759 ± 0.0484 | 0.6710 ± 0.0341 | 0.7663 ± 0.0318 | 0.7338 ± 0.0357 |
| **GPC** | (m)AUC | 0.7081 ± 0.0317 | 0.5826 ± 0.0289 | 0.5532 ± 0.0634 | 0.5779 ± 0.0242 | 0.5755 ± 0.0053 |
| **SSL-GPC** | (m)AUC | 0.8986 ± 0.0469 | 0.7085 ± 0.0463 | 0.7120 ± 0.0768 | 0.7105 ± 0.0207 | 0.6844 ± 0.0227 |
| **RFC** | (m)AUC | 0.9026 ± 0.0319 | 0.7417 ± 0.0714 | 0.6990 ± 0.0837 | 0.7089 ± 0.0248 | 0.6917 ± 0.0236 |
| **SSL-RFC** | (m)AUC | 0.8985 ± 0.0327 | 0.7515 ± 0.0625 | 0.6991 ± 0.0780 | 0.7217 ± 0.0311 | 0.7153 ± 0.0250 |

**Summary**:
* Applying SSL consistently improved performance across most tasks.
* **SSL-SVC** and **SSL-GPC** showed significant improvements, particularly in the challenging **sMCI vs. pMCI** classification.

---

## 💻 Environment

This project is implemented using Python 3.x.
* **PyTorch** (for SimCLR & Deep Learning)
* **scikit-learn** (for Downstream Classifiers)
* **NumPy**

---

## 📖 Citation

If you find this work useful, please cite our paper:

**Improving Machine Learning-based Early Diagnosis Framework of Degenerative Brain Disease through Self-Supervised Learning**
> KIISE Korea Computer Congress 2025 (KCC 2025), pp. 2105–2107
> [🔗Paper](https://drive.google.com/file/d/1xLmii0rHMxKl9XijAUx8BxWK54Muccz1/view?usp=sharing)

```bibtex
@inproceedings{cho2025improving,
  title={Improving Machine Learning-based Early Diagnosis Framework of Degenerative Brain Disease through Self-Supervised Learning},
  author={Cho, Minseo and Park, Chanmi and Kim, Yeonji and Ko, Wonjun},
  booktitle={Korea Computer Congress 2025 (KCC 2025)},
  pages={2105--2107},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This work was supported by the **National Research Foundation of Korea (NRF)** grant funded by the Korea government (MSIT) (No. RS-2025-00519583).
