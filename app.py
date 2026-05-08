import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
from utils.model import efficientnet_model
import streamlit as st

# set the title
st.title('Face Problem Classification')

# create the class names
class_names = ['Blackheads', 'Cyst', 'Papules', 'Pustules', 'Whiteheads']

# Load the model
model = efficientnet_model(num_classes=len(class_names))
model.load_state_dict(torch.load("models/effnetb2_model_face_problem.pth", map_location=torch.device('cpu')))

# Define the image transforms
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create a file uploader
uploader_file = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])

# Condition if the file is uploaded
if uploader_file is not None:

    #  Open the Image
    image = Image.open(uploader_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    image_tensor = transforms(image).unsqueeze(0)

    # Make predictions
    model.eval()
    with torch.inference_mode():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(prob, dim=1).item()

    # Display the prediction class
    st.write(f'Predicted Class: {class_names[predicted_class]} \nProbability: {prob[0][predicted_class]:.4f}')

# Create session state for camera input
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Create condition for camera input
if st.button('Take a picture'):
    st.session_state.show_camera = True

# Create condition if the camera input is true
if st.session_state.show_camera:

    # Capture the image from camera
    foto = st.camera_input('Take a Picture')

    # Condition if the image is captured
    if foto is not None:

        # Open the captured image
        foto = Image.open(foto)

        # Display the capture image
        st.image(foto, caption='Captured Image', use_container_width=True)

        # Preprocess the captured image
        foto_tensor = transforms(foto).unsqueeze(0)

        # Make predictions
        model.eval()
        with torch.inference_mode():
            output = model(foto_tensor)
            prob = torch.softmax(output, dim=1)
            prediction_class = torch.argmax(prob, dim=1).item()

        # Display the prediction class
        st.write(f'Predicted Class: {class_names[prediction_class]} \nProbability: {prob[0][prediction_class]:.4f}')

    # Create a button to close the camera input
    if st.button('Close Camera'):
        st.session_state.show_camera = False