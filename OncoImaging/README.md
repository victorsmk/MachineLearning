# OncoImaging

## Overview

OncoImaging is a project demonstrating a multi-class, CNN-based approach for cancer detection using histopathological tissue images. This project implements and compares two distinct models:

1.  A **Transfer Learning Model** using the pre-trained MobileNetV3Large architecture.
2.  A **Custom CNN Model** built from scratch.

Both models are trained to classify tissue images into 7 different types of cancer.

---

## Dataset

The project uses the **Multi Cancer Image (MCI)** dataset available on Kaggle.

* **Link:** https://www.kaggle.com/datasets/obulisainaren/multi-cancer/data
* **Note:** The "ALL" class, which is an aggregation of all cancer types, was programmatically excluded from the analysis to focus on classifying distinct cancer types.

---

## Requirements

* `tensorflow` (and `keras`)
* `numpy`
* `seaborn`
* `scikit-learn`
* `matplotlib`

---

## Project Walkthrough

This document outlines the entire process, from data loading to model training and evaluation.

### 1. Data Loading and Preparation
Loading multi-cancer dataset excluding ALL Found 110002 files belonging to 7 classes. Using 82502 files for training. Loading multi-cancer dataset excluding ALL Found 110002 files belonging to 7 classes. Using 27500 files for validation.
### 2. Data Visualization

To understand the data, a sample image from each of the 7 classes was plotted.



### 3. Model 1: Transfer Learning with MobileNetV3Large

The first model used transfer learning. The pre-trained **MobileNetV3Large** model (with "imagenet" weights) was used as the base, with its layers frozen. Custom classification layers were added on top:

* GlobalAveragePooling2D
* Dense (1024 units, relu)
* Dense (512 units, relu)
* Dense (128 units, relu)
* Dense (7 units, softmax)

The model was compiled with the Adam optimizer (learning rate 0.0001) and trained for 5 epochs.

#### Training and Evaluation (Model 1)

The model trained in approximately 1 hour and 15 minutes and achieved high performance.

**Epoch 5/5:** `accuracy: 0.9998 - loss: 9.5385e-04 - val_accuracy: 0.9996 - val_loss: 0.0012`

**Final Evaluation Results:**
* **Loss:** 0.00116
* **Accuracy:** 0.99956

#### Results (Model 1)




---

### 4. Model 2: Custom CNN

The second model was a custom Convolutional Neural Network built from scratch. Before training, the data was normalized using a `tf.keras.layers.Rescaling(1./255)` layer.

#### Model Architecture

The custom model consists of several stacked Conv2D, MaxPool2D, and Dropout layers, followed by a Flatten layer and Dense layers for classification. The model has approximately 3.4 million trainable parameters.
#### Training and Evaluation (Model 2)

* **Compiler:** Adam optimizer (learning rate 0.0001)
* **Data Augmentation:** Random flips, brightness, and contrast adjustments were applied to the training data.
* **Callbacks:** `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` were used to manage the training process and save the best model.

The model was trained for 5 epochs. As predicted, this took a very long time to train—about **10 hours**. However, the validation accuracy and loss are comparable to the pre-trained model.

**Epoch 5/5:** `accuracy: 0.9917 - loss: 0.0302 - val_accuracy: 0.9932 - val_loss: 0.0227`

#### Results (Model 2)




---

### 5. Conclusion

So why did I make two models then? Well for one, I wanted to have a
finished project even if it meant that I didn't have a model built on my
own, and the second reason would be that after finding out there were
pre-trained CNN models, I was really curious on how well they would
perform.

Both models show the power of CNNs for image classification, achieving very high accuracy on this dataset.

---

### 6. Future Work

This project is merely a speck of what I had in mind. Initially, I didn't want
to use tissue images, because they are very hard to obtain from the
human body - expensive, intrusive, and slow.

My initial idea was to use **SERS (Surface-Enhanced Raman Spectroscopy) spectrograms**.
This data is much easier to obtain, requiring only bodily fluids to get
information about the state of the whole body.



Unfortunately, I've searched a lot and there doesn't seem to be any
publicly available labeled datasets for SERS, so for now I had to settle
with what was shown above. This will probably be the next thing that I'll work on, so if
that sounds interesting to you stay tuned!

A custom function was written to load the dataset, specifically to exclude the "ALL" class directory. The remaining 7 classes were loaded, resulting in a total of 110,002 images. The data was split into a **training set (82,502 files)** and a **validation set (27,500 files)** using a 75/25 split.
