# Multimodal Emotion Regression with BEiT-3

## Project Overview

This project performs **multimodal emotion regression** from images and text, predicting continuous affective signals: **valence** (pleasantness) and **arousal** (excitement). Unlike traditional classification approaches (e.g., Ekman’s six emotions), this method captures **fine-grained emotional cues** suitable for real-world applications. The model leverages a **multimodal transformer (BEiT-3)** to jointly process visual and textual inputs.

---

## Background

This work builds upon a previous project on multilingual valence-arousal prediction from text:  
[gmendes9/multilingual_va_prediction](https://github.com/gmendes9/multilingual_va_prediction?tab=readme-ov-file)

The original approach focused on **unimodal text-based emotion regression** using multilingual transformer models.

In this project, we extend the problem along four main **aspects**:

- **Model architecture**: we adopt a multimodal transformer (**BEiT-3**) to jointly process visual and textual inputs, enabling cross-modal representation learning.  
- **Multimodal integration**: moving from unimodal text modeling to a **vision-language setting**, allowing the model to capture complementary emotional cues from both images and text.  
- **Multilingual alignment**: to support multilingual textual inputs, **cross-lingual embedding alignment** is performed using the **VecMap** framework [Artetxe et al., 2016](https://github.com/artetxem/vecmap), projecting textual representations into a shared semantic space.  
- **Uncertainty-aware prediction (planned)**: inspired by [Carlier et al., 2025](https://arxiv.org/pdf/2501.18991.pdf), we plan to implement an **Optimal Transport-based conformal prediction** framework to quantify prediction uncertainty and generate confidence intervals for valence-arousal outputs.

---

## Model Architecture

The architecture consists of:

1. **BEiT-3 backbone**, pretrained on VQA data at 480px resolution for vision-language understanding.  
2. **Custom regression head**:
   - Shared **pooler** over the joint BEiT-3 embeddings  
   - Two **independent MLP branches**, one for **valence** and one for **arousal**  

The model enables continuous emotion prediction from **multimodal inputs**, providing fine-grained affective representations.

---

## Data and Fine-Tuning

We perform fine-tuning on a large collection of datasets (~50 in total), including **multimodal (text + image), text-only, and image-only datasets**. Some of the most notable datasets are:

- **AffectNet**
- **AffWild2**
- **ArtELingo**
- **MELD**
- **EmoBank**
- **GoEmotions**
- **EMOTIC**
- **IEMOCAP** 

To select the most effective loss function for final fine-tuning, I conducted a preliminary experiment on a **stratified subset of 1% of the dataset (~26k samples)** for 7 epochs. We evaluated five candidate losses:

- Mean Squared Error (MSE)  
- Robust loss  
- Concordance Correlation Coefficient Loss (CCCL)  
- MSE + CCCL  
- Robust loss + CCCL  

Among the candidate loss functions, **MSE achieved the lowest RMSE** and was therefore selected as the objective for the final valence-arousal regression on the full dataset.


