#!/usr/bin/env python
# coding: utf-8

"""
Food-101 Classification and USDA FoodData Central Preprocessing Script
This script preprocesses the Food-101 dataset for image classification, performs hyperparameter tuning, 
trains a fine-tuned model, and preprocesses the USDA FoodData Central dataset for downstream tasks.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import Hyperband

# --------------------------------------------------------------
# 1. Food-101 Dataset Preprocessing
# --------------------------------------------------------------

# Define paths
FOOD101_BASE_PATH = '/Users/shubhamgaur/Desktop/NU/Sem4/Gen AI/Project/Dataset/food-101/images/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Load training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    FOOD101_BASE_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    FOOD101_BASE_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

# Prefetch for performance optimization
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Define image augmentation pipeline
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# --------------------------------------------------------------
# 2. Hyperparameter Tuning for Food-101 Classification
# --------------------------------------------------------------

def build_model(hp):
    base_model = EfficientNetB7(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model = Sequential([
        data_augmentation,
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(hp.Float('dropout_rate', 0.3, 0.6, step=0.1)),
        Dense(hp.Int('dense_units', 256, 1024, step=128), activation="relu"),
        Dropout(hp.Float('dropout_rate_2', 0.2, 0.5, step=0.1)),
        Dense(101, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 3e-4, 1e-3])),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Tuner setup
tuner = Hyperband(
    build_model, objective="val_accuracy", max_epochs=10, factor=3,
    directory="tuning_dir", project_name="food101_tuning"
)
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

# Uncomment to perform hyperparameter tuning
# tuner.search(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping])
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Hardcoded hyperparameters based on previous tuning
best_hyperparams = {
    "dropout_rate": 0.4,
    "dense_units": 1024,
    "dropout_rate_2": 0.3,
    "learning_rate": 0.0003
}

# --------------------------------------------------------------
# 3. Train the Best Model
# --------------------------------------------------------------

def build_model_with_best_hps():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(best_hyperparams["dropout_rate"]),
        Dense(best_hyperparams["dense_units"], activation="relu"),
        Dropout(best_hyperparams["dropout_rate_2"]),
        Dense(101, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyperparams["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

best_model = build_model_with_best_hps()

history = best_model.fit(
    train_ds, validation_data=val_ds, epochs=15,
    callbacks=[early_stopping]
)

best_model.save("food101_best_model_finetuned.keras")

# Evaluate the model
loss, accuracy = best_model.evaluate(val_ds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Visualize training progress
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Model Loss")
plt.show()

# --------------------------------------------------------------
# 4. USDA FoodData Central Preprocessing
# --------------------------------------------------------------

USDA_INPUT_FILE = "/Users/shubhamgaur/Desktop/NU/Sem4/Gen AI/Project/Dataset/USDA FoodData/fda_approved_food_items_w_nutrient_info.csv"
USDA_OUTPUT_FILE = "/Users/shubhamgaur/Desktop/NU/Sem4/Gen AI/Project/Dataset/USDA FoodData/cleaned_food_data_v2.csv"

# Load the dataset
data = pd.read_csv(USDA_INPUT_FILE)
data = data.dropna(axis=1, thresh=len(data) * 0.5).fillna(0)

# Rename columns
data = data.rename(columns={
    "fdc_id": "FDC_ID",
    "brand_owner": "Brand",
    "description": "Description",
    "ingredients": "Ingredients",
    "gtin_upc": "UPC",
    "serving_size": "ServingSize",
    "serving_size_unit": "ServingUnit",
    "branded_food_category": "FoodCategory",
    "modified_date": "ModifiedDate",
    "available_date": "AvailableDate",
    "Energy-KCAL": "Calories",
    "Protein-G": "Protein",
    "Total lipid (fat)-G": "Fat",
    "Carbohydrate, by difference-G": "Carbohydrates"
})

# Convert data types
data["ModifiedDate"] = pd.to_datetime(data["ModifiedDate"], errors="coerce")
data["AvailableDate"] = pd.to_datetime(data["AvailableDate"], errors="coerce")
numeric_columns = ["ServingSize", "Calories", "Protein", "Fat", "Carbohydrates"]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)

# Remove duplicates
data = data.drop_duplicates()

# Feature engineering
data["Calories_per_gram"] = data["Calories"] / data["ServingSize"]
data["Protein_Ratio"] = data["Protein"] / (data["Protein"] + data["Fat"] + data["Carbohydrates"])
data["Fat_Ratio"] = data["Fat"] / (data["Protein"] + data["Fat"] + data["Carbohydrates"])
data["Carbs_Ratio"] = data["Carbohydrates"] / (data["Protein"] + data["Fat"] + data["Carbohydrates"])

# Normalize numeric columns
scaler = MinMaxScaler()
data[["Calories", "Protein", "Fat", "Carbohydrates"]] = scaler.fit_transform(
    data[["Calories", "Protein", "Fat", "Carbohydrates"]]
)

# Save the cleaned dataset
data.to_csv(USDA_OUTPUT_FILE, index=False)
print(f"Cleaned USDA FoodData saved to {USDA_OUTPUT_FILE}")
