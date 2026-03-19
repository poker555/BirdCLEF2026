
import torch 
import torch.nn as nn
import timm

class BirdModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=234):
        super(BirdModel,self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1
        )

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features,num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ =='__main__':
    model = BirdModel()
    dummy_input = torch.randn(8, 1, 128, 313)
    output = model(dummy_input)
    print("預測模型結果",output.shape)