import streamlit as st
from PIL import Image
import tensorflow as tf
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key securely from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set title for the Streamlit app
st.title("NutriVision: Food Nutrient Analysis")

# ====================
# Function Definitions
# ====================

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the model's input requirements.
    """
    # Convert image to RGB to handle images with alpha channels
    image = image.convert("RGB")
    # Resize to the model's input size
    image = image.resize((224, 224))
    # Convert to NumPy array and normalize
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    return image


@st.cache_resource
def load_food101_model():
    """
    Load the pre-trained Food-101 model and cache it for faster access.
    """
    model_path = './models/food101_model.keras'
    model = tf.keras.models.load_model(model_path)
    return model


def search_usda(predicted_food, usda_data):
    """
    Search the USDA FoodData Central dataset for the predicted food.
    """
    matches = usda_data[
        usda_data["Description"].str.contains(predicted_food, case=False, na=False)
        | usda_data["FoodCategory"].str.contains(predicted_food, case=False, na=False)
    ]
    return matches


@st.cache_resource
def load_usda_data():
    """
    Load and cache the cleaned USDA FoodData Central dataset.
    """
    dataset_path = './Dataset/cleaned_food_data.csv'
    usda_data = pd.read_csv(dataset_path)
    return usda_data


@st.cache_resource
def load_food_classes():
    """
    Load and cache the Food-101 class labels.
    """
    classes_path = './Dataset/food-101/meta/classes.txt'
    with open(classes_path, "r") as file:
        food_classes = file.read().splitlines()
    return food_classes


def generate_dietary_insight(nutrition_text):
    """
    Use GPT-4 to generate dietary insights based on nutritional data.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful nutrition assistant."},
                {"role": "user", "content": f"Based on this data: {nutrition_text}, provide dietary advice."}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating dietary insight: {e}"



# =========================
# Load Models and Datasets
# =========================

# Load the Food-101 model
model = load_food101_model()

# Load the USDA dataset
usda_data = load_usda_data()

# Load Food-101 class labels
food_classes = load_food_classes()

# =================
# Streamlit Workflow
# =================

# Step 1: Upload image
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Analyzing the image...")

        # Step 2: Preprocess and predict food class
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = prediction.argmax(axis=-1)[0]
        predicted_food = food_classes[predicted_class_index]
        st.write(f"Predicted Food: {predicted_food}")

        # Step 3: Search USDA dataset for nutritional info
        matches = search_usda(predicted_food, usda_data)
        if not matches.empty:
            top_match = matches.iloc[0]
            st.write("Nutritional Information:")
            st.write(f"Calories: {top_match['Calories']} kcal")
            st.write(f"Protein: {top_match['Protein']} g")
            st.write(f"Fat: {top_match['Fat']} g")
            st.write(f"Carbohydrates: {top_match['Carbohydrates']} g")

            # Step 4: Generate dietary insights using GPT-4
            nutrition_text = (
                f"This {predicted_food} contains {top_match['Calories']} kcal, "
                f"{top_match['Protein']} g protein, {top_match['Fat']} g fat, "
                f"and {top_match['Carbohydrates']} g carbohydrates."
            )
            dietary_insight = generate_dietary_insight(nutrition_text)
            st.write("Dietary Insight:")
            st.write(dietary_insight)
        else:
            st.write("No matching food item found in the USDA dataset.")

    except Exception as e:
        st.error(f"Error: {e}")
