import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.serialization  # Add this import for safe loading

# Define SimpleCNN again (you need this class for the model to be correctly loaded)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.pool2 = nn.MaxPool2d(2, 2)  # Adding another pooling layer
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128*64*64, 38)  # Number of output classes

    def forward(self, X):
        X = self.conv1(X)
        X = self.pool1(X)
        X = self.conv2(X)
        X = self.pool2(X)  # Second pooling layer
        X = X.view(-1, 128*64*64)  # Flatten the tensor
        X = self.fc1(X)
        X = self.dropout(X)
        return X

# Safe load configuration
torch.serialization.add_safe_globals([SimpleCNN])  # Allow SimpleCNN class during model loading

# Load the whole model (includes architecture and weights)
model = torch.load('mymodelnoaugwhole_model.pth', weights_only=False)  # Load with weights_only=False
model.eval()  # Set model to evaluation mode

# Transformation for the image (resize, normalization, etc.)
transform=transforms.ToTensor() 

# Class names (replace with your dataset's labels)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
               'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus']  # Replace with your actual class names

# Function to predict the class of an image
def predict_image(uploaded_image):
    # Open the image using PIL
    image = Image.open(uploaded_image)

    # Apply the transformation and add batch dimension
    image = transform(image).unsqueeze(0)  # Add batch dimension (1 image in the batch)
    
    # Get the model's prediction
    with torch.no_grad():  # No need to track gradients for inference
        output = model(image)
        
    # Get the index of the class with the highest score
    _, predicted_idx = torch.max(output, 1)
    
    # Map the predicted index to the class label
    return class_names[predicted_idx.item()]  # Return the class name

# Streamlit UI
st.title('Plant Disease Prediction')
st.write('Upload a plant image to predict its disease class.')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    prediction = predict_image(uploaded_image)
    
    # Display the prediction result
    st.write(f"Prediction: {prediction}")
