# Multimodal Emotion Regression with BEiT-3

## Project Overview

This project extends BEiT-3 for **multimodal emotion regression**, predicting continuous affective dimensions — **valence** and **arousal** — from **image-text inputs**. The goal is to move beyond classification-based approaches and model fine-grained emotional signals, which are more suitable for **real-world and clinical applications**.

The project leverages pretrained vision-language models and integrates textual multilingual inputs, producing a unified framework for **multimodal and multilingual emotion regression**.

---

## Background

This work builds upon a previous project on multilingual valence-arousal prediction from text:  
[gmendes9/multilingual_va_prediction](https://github.com/gmendes9/multilingual_va_prediction?tab=readme-ov-file)

The original approach focused on **unimodal text-based emotion regression** using multilingual transformer models.

In this project, we extend the problem along three main dimensions:

- **Model architecture**: we adopt a multimodal transformer (**BEiT-3**) to jointly process visual and textual inputs, enabling cross-modal representation learning.  
- **Multimodal integration**: moving from unimodal text modeling to a **vision-language setting**, allowing the model to capture complementary emotional cues from both images and text.  
- **Multilingual alignment**: to support multilingual inputs, **cross-lingual embedding alignment is performed using VecMap**, projecting textual representations into a shared semantic space.

This results in a **unified framework** for multimodal and multilingual emotion regression.

---

## Model Architecture

The architecture consists of:

1. **BEiT-3 backbone** pretrained on VQA data at 480px resolution for vision-language understanding.  
2. **Custom regression head**:
   - Shared **pooler** over the joint BEiT-3 embeddings  
   - Two **independent MLP branches**, one for **valence** and one for **arousal**  

The model allows continuous emotion prediction from **multimodal inputs**, providing fine-grained affective representations.

---

## Data & Fine-tuning

- We perform **fine-tuning on ~10% of the dataset** (approximately 400k samples, stratified).  
- **Loss function selection** was empirically evaluated among:
  - Mean Squared Error (MSE)  
  - Robust loss  
  - Concordance Correlation Coefficient Loss (CCCL)  
  - MSE + CCCL  
  - Robust loss + CCCL  

This preliminary screening identifies the most effective objective for valence-arousal regression.

---

## Multilingual Preprocessing

- Textual inputs are aligned across languages using **VecMap**, projecting all embeddings into a **shared semantic space**.  
- This enables BEiT-3 to handle multilingual textual cues alongside visual information.

---

## Setup

Clone the repo and install required packages:

```bash
git clone https://github.com/manueldg1/beit3-emotion-regression.git
cd beit3-emotion-regression
pip install -r requirements.txt
