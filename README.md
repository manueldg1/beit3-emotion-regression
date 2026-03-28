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

## Data and Fine-Tuning

We perform fine-tuning on multimodal datasets combining images and textual descriptions.

For selecting the most effective loss function for the final fine-tuning, we conducted a preliminary experiment on a stratified subset of ~10% of the dataset (approximately 400k samples). We evaluated the performance of five candidate losses:

- Mean Squared Error (MSE)
- Robust loss
- Concordance Correlation Coefficient Loss (CCCL)
- MSE + CCCL
- Robust loss + CCCL

This screening allowed us to empirically identify the most suitable objective for valence-arousal regression before training on the full dataset.

---

## Multilingual Preprocessing

- Textual inputs are aligned across languages using **VecMap**, projecting all embeddings into a **shared semantic space**.  
- This enables BEiT-3 to handle multilingual textual cues alongside visual information.

---

