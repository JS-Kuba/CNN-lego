import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNN
from data import DataHandler

# Load the saved model
model = CNN()
model.load_state_dict(torch.load("final_models/model_05_12_19_33_57.pth"))
model.eval()

nv = DataHandler.get_normalization_variables()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(nv["mean"], nv["std"]),
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

image_path = "./IMG_0291.jpg"
predicted_class = predict_image(image_path)
print("Predicted class:", predicted_class)
