# densenet.py
import torch
import torch.nn as nn
from torchvision import models

def initialize_densenet(num_classes):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
