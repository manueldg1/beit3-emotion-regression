# Multimodal Emotion Regression with BEiT-3

## Project Overview

The analysis of emotions expressed in textual documents and/or images has numerous practical applications. The task is commonly modeled as a classification problem, representing affective states (e.g., Ekman’s six basic emotions) as specific classes and using Transformer-based models to represent the inputs and perform the classification. Alternatively, one can consider approaches based on **dimensional emotion analysis**, focused on rating emotions according to a pre-defined set of dimensions and offering a more nuanced way to distinguish between different affective states. In this case, the emotions are represented in a continuous numerical space, with the most common dimensions defined as **valence** and **arousal**. In particular, valence describes the pleasantness of a stimulus, ranging from negative to positive feelings, while arousal represents the degree of excitement provoked by a stimulus, from calm to excited. Apart from a few exceptions, the development of regression models for dimensional emotion analysis has been less studied in the literature.

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
