import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from baal.active import FileDataset, ActiveLearningDataset
from glob import glob
from baal.utils.metrics import Accuracy

# Define data directory and transformations
data_dir = 'data'
classes = os.listdir(data_dir)
files = glob(os.path.join(data_dir, '*/*.jpg'))
print(classes)

data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# split dataset
train, test = train_test_split(files, random_state=1337, test_size=0.25)  # Split 75% train, 25% validation
print(f"Train: {len(train)}, Valid: {len(test)}, Num. classes: {len(classes)}")

# Use the custom image loading function in your dataset initialization
train_dataset = FileDataset(train, [-1] * len(train), data_transform)
test_dataset = FileDataset(test, [-1] * len(test), test_transform)


active_learning_ds = ActiveLearningDataset(train_dataset, pool_specifics={'transform': data_transform})


# Create DataLoader objects for train and test sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Check the maximum pixel value of the first batch
images, labels = next(iter(train_loader))
print(images.max())

# Visualize a few images from the batch
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    ax[idx].imshow(images[idx].permute(1, 2, 0))  # PyTorch uses CxHxW, so permute for imshow
    ax[idx].set_title(labels[idx].item())
plt.show()



# model --->


import torch
from torch import nn, optim
from torchsummary import summary
from baal.modelwrapper import ModelWrapper
from torchvision.models import vgg16
from baal.bayesian.dropout import MCDropoutModule

USE_CUDA = torch.cuda.is_available()

model = vgg16(weights=None, num_classes=len(classes))

model = MCDropoutModule(model)
if USE_CUDA:
  model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

input_size = (3, 255, 255)
summary(model, input_size=input_size)

# ModelWrapper is an object similar to keras.Model.
baal_model = ModelWrapper(model, criterion)

from baal.active.heuristics import BALD
heuristic = BALD(shuffle_prop=0.1)

def get_label(img_path):
  return classes.index(img_path.split('/')[-2])

for idx in range(len(test_dataset)):
  img_path = test_dataset.files[idx]
  test_dataset.label(idx, get_label(img_path))
  
  
# Let's label 100 training examples randomly first.
# Note: the indices here are relative to the pool of unlabelled items!
train_idxs = np.random.permutation(np.arange(len(train_dataset)))[:100].tolist()
labels = [get_label(train_dataset.files[idx]) for idx in train_idxs]
active_learning_ds.label(train_idxs, labels)

print(f"Num. labeled: {len(active_learning_ds)}/{len(train_dataset)}")

# plotting loss curve and calculating evalutaion matrics

baal_model.add_metric(name='accuracy',initializer=lambda : Accuracy())
metrics_data = baal_model.get_metrics()

train_accuracy_list = []
test_accuracy_list = []
train_losses = []
test_losses = []

# 2. Train the model for a few epoch on the training set.
train_loss=baal_model.train_on_dataset(active_learning_ds, optimizer, batch_size=8, epoch=5, use_cuda=USE_CUDA)
test_loss=baal_model.test_on_dataset(test_dataset, batch_size=8, use_cuda=USE_CUDA)

train_losses.append(train_loss)
test_losses.append(test_loss)

metrics_data=baal_model.get_metrics()

test_accuracy_list.append(metrics_data['test_accuracy'])
train_accuracy_list.append(metrics_data['train_accuracy'])


print("Metrics:", {k:v.avg for k,v in baal_model.metrics.items()})

pool = active_learning_ds.pool
if len(pool) == 0:
  raise ValueError("We're done!")


# We make 15 MCDropout iterations to approximate the uncertainty.
predictions = baal_model.predict_on_dataset(pool, batch_size=8, iterations=15, use_cuda=USE_CUDA, verbose=False)
# We will label the 10 most uncertain samples.
top_uncertainty = heuristic(predictions)[:10]


oracle_indices = active_learning_ds._pool_to_oracle_index(top_uncertainty)
labels = [get_label(train_dataset.files[idx]) for idx in oracle_indices]
print(list(zip(labels, oracle_indices)))
active_learning_ds.label(top_uncertainty, labels)



# 5. If not done, go back to 2.
for step in range(5): # 5 Active Learning step!
  # 2. Train the model for a few epoch on the training set.
  print(f"Training on {len(active_learning_ds)} items!")
  train_loss=baal_model.train_on_dataset(active_learning_ds, optimizer, batch_size=8, epoch=5, use_cuda=USE_CUDA)
  test_loss=baal_model.test_on_dataset(test_dataset, batch_size=8, use_cuda=USE_CUDA)

  train_losses.append(train_loss)
  test_losses.append(test_loss)
  
  metrics_data=baal_model.get_metrics()
  
  test_accuracy_list.append(metrics_data['test_accuracy'])
  train_accuracy_list.append(metrics_data['train_accuracy'])
  
  print("Metrics:", {k:v.avg for k,v in baal_model.metrics.items()})
  
  # 3. Select the K-top uncertain samples according to the heuristic.
  pool = active_learning_ds.pool
  if len(pool) == 0:
    print("We're done!")
    break
  predictions = baal_model.predict_on_dataset(pool, batch_size=8, iterations=15, use_cuda=USE_CUDA, verbose=False)
  top_uncertainty = heuristic(predictions)[:10] 
  # 4. Label those samples.
  oracle_indices = active_learning_ds._pool_to_oracle_index(top_uncertainty)
  labels = [get_label(train_dataset.files[idx]) for idx in oracle_indices]
  active_learning_ds.label(top_uncertainty, labels)
  
  
  
  
  torch.save({
  'active_dataset': active_learning_ds.state_dict(),
  'model': baal_model.state_dict(),
  'metrics': {k:v.avg for k,v in baal_model.metrics.items()}
}, '/mnt/c/Users/yoshi/VPROJECTS/DL PROJECT/model/brain_active_classifiaction_on fulldataset.pth')
  
  

# Plotting
# Define the epochs based on the length of the lists
epochs = range(1, len(train_losses) + 1)

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'r', label='Training Loss')
plt.plot(epochs, test_losses, 'b--', label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot1.png')  # Save the loss plot as loss_plot.png
plt.show()

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy_list, 'r', label='Training Accuracy')
plt.plot(epochs, test_accuracy_list, 'b--', label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot1.png')  # Save the accuracy plot as accuracy_plot.png
plt.show()

