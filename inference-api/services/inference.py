import torch.nn as nn
import timm
from huggingface_hub import PyTorchModelHubMixin

class BigFiveRegressor(nn.Module, PyTorchModelHubMixin):
    def __init__(self, timm_name, use_complex_head=True):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0) 
        num_features = self.backbone.num_features
        
        if use_complex_head:
            self.regression_head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 5),
                nn.Sigmoid()
            )
        else:
            self.regression_head = nn.Sequential(
                nn.Linear(num_features, 5),
                nn.Sigmoid()
            )
            
    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features)
