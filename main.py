import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Define the classify function before it is used
def classify(image, model, class_names):
    # Function to classify the image
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Set the title of the app
st.title('Pneumonia Classification and  Detection' )

# Load and display an image under the title but above the file uploader
image_path = r'C:\Users\Deepu\Downloads\Screenshot 2024-10-10 162404.png'  
image = Image.open(image_path)
st.image(image, caption="Lung X-ray Example", use_column_width=True)  

# Load the classifier model
model = load_model(r'C:\Users\Deepu\Downloads\pneumonia_classifier.h5')

# Load class names
with open(r'C:\Users\Deepu\Downloads\labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Create a file uploader widget
file = st.file_uploader('Please upload a chest X-ray image', type=['jpeg', 'jpg', 'png'])

# Initialize variable to hold the uploaded image
uploaded_image = None

# Check if a file has been uploaded
if file is not None:
    try:
        # Open and resize the uploaded image
        uploaded_image = Image.open(file).convert('RGB')
        
        # Resize the image (e.g., 300x300 pixels, adjust as necessary)
        resized_image = uploaded_image.resize((100, 100))

        # Display the resized image
        st.image(resized_image, caption="Resized Uploaded Image", use_column_width=False)

        # Classify the original image (since resizing might not be required for classification)
        class_name, conf_score = classify(uploaded_image, model, class_names)

        # Display the classification result
        st.write("## Predicted Class: {}".format(class_name))
        st.write("### Confidence Score: {}%".format(int(conf_score * 1000) / 10))

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
