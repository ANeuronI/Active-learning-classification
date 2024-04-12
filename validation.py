import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision.models as models
from torchvision.models import vgg16
from PIL import Image
from sklearn.metrics import accuracy_score

data_dir = 'data'
classes = os.listdir(data_dir)
input_image_path = '/mnt/c/Users/yoshi/VPROJECTS/DL PROJECT/pred/pred49.jpg'

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the input image
input_image = Image.open(input_image_path)
input_tensor = data_transform(input_image)
input_batch = input_tensor.unsqueeze(0)

# Move input tensor to the appropriate device
# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_batch = input_batch.to(device)
# Move the input tensor to the cpu
# device = torch.device('cpu')
# input_batch = input_batch.to(device)


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

pathofpretrain = '/mnt/c/Users/yoshi/VPROJECTS/DL PROJECT/model/brain_active_classifiaction_on_big.pth'
num_classes = len(classes)  
model = CustomVGG16(num_classes)
model.to(device)

checkpoint = torch.load(pathofpretrain)['model']
state_dict = {k.replace("parent_module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)
model.eval()

# Now you can use the model for prediction
# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class probabilities
probabilities = torch.softmax(output, dim=1)[0]

# Get the index of the class with the highest probability
predicted_class_index = torch.argmax(probabilities).item()

# Get the predicted class label
predicted_class = classes[predicted_class_index]

# Save the predicted image with the predicted class label appended to its filename
# predicted_image_name = os.path.splitext(os.path.basename(input_image_path))[0] + '_' + predicted_class + '.jpg'
# predicted_image_path = os.path.join(os.path.dirname(input_image_path), predicted_image_name)
# input_image.save(predicted_image_path)

# Print the predicted class label and the path to the predicted image
print("Predicted Class:", predicted_class)
# print("Predicted Image Path:", predicted_image_path)

