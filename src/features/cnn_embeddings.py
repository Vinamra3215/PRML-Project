import numpy as np
import torch
from PIL import Image
from src.features.base import FeatureExtractor


class CNNEmbeddingExtractor(FeatureExtractor):

    def __init__(self, backbone: str = "resnet50", device: str = "cuda"):
        import timm
        from torchvision import transforms

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
