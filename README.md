# 🌿 Plant Disease Classification Web Application

A Flask-based deep learning application that classifies plant leaf images into various disease categories (or healthy), providing actionable insights and disease descriptions to help farmers, gardeners, and agricultural experts.

[![Watch the demo video](cover.jpg)](https://www.youtube.com/watch?v=3IMj5gWH3sc)

## 📋 Features

- 📤 Upload plant leaf images for instant analysis
- 🤖 Real-time classification using a trained deep learning model (e.g., MobileNetV2 or ResNet50)
- 📊 Visual probability distribution for all possible classes
- 🧾 Disease-specific interpretation and management suggestions
- 📱 Fully responsive UI for desktop and mobile use

## 🔍 Supported Plant Classes

Includes diseases and healthy states for:
- 🍎 Apple (Scab, Black Rot, Cedar Apple Rust, Healthy)
- 🌽 Corn (Common Rust, Northern Leaf Blight, etc.)
- 🍇 Grape, 🍅 Tomato, 🍓 Strawberry, 🥔 Potato, 🫑 Pepper, 🍑 Peach, 🍊 Orange, and more...

> Refer to the app's result page for detailed disease-specific information.

## 🚀 Getting Started

### Prerequisites

Make sure the following are installed:
- Python 3.7 or higher
- pip (Python package manager)
- A trained model file named `plant_disease_model.h5` placed in the `model/` directory

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/plant-disease-classifier.git
cd plant-disease-classifier

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
