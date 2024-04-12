import os
import numpy as np
from torchvision import transforms
from torch import nn
import torchvision.models as models
from PIL import Image
import torch
import gradio as gr


# Define the data transformation
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the pre-trained model
class CustomVGG16(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the trained model and classes
data_dir = 'data'
classes = os.listdir(data_dir)
num_classes = len(classes)  
model = CustomVGG16(num_classes)

# Load the model weights
pathofpretrain = '/mnt/c/Users/yoshi/VPROJECTS/DL PROJECT/model/brain_active_classifiaction_on_big.pth'
checkpoint = torch.load(pathofpretrain)['model']
state_dict = {k.replace("parent_module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)
model.eval()

# Function to predict image class
def classify_image(input_image):
    input_tensor = data_transform(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class = classes[predicted_class_index]

    return predicted_class

# Define Gradio interface
input_image = gr.Image(type="pil")
output_text = gr.Label(num_top_classes=2)

# Create the Gradio app
gr.Interface(fn=classify_image, inputs=input_image, outputs=output_text, title="Brain Tumor Classification ACTIVE LEARNING").launch(share=True)
