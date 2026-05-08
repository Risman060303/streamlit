import torch
from torch import nn
import torchvision

def efficientnet_model(num_classes: int):

  # 1. 2 Get the pre-trained model
  weight = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  model = torchvision.models.efficientnet_b2(weights=weight)

  # 3. Freeze the classifier head
  for param in model.parameters():
    param.requires_grad = False

  # (classifier): Sequential(
  #   (0): Dropout(p=0.3, inplace=True)
  #   (1): Linear(in_features=1408, out_features=1000, bias=True)
  # )

  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1408, out_features=500),
      nn.ReLU(),
      nn.Linear(in_features=500, out_features=num_classes)
  )

  return model
