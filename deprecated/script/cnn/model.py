import torch
import torch.nn as nn
import timm

NUM_CLASSES = 234   # 物種數
NUM_GROUPS  = 5     # 大類數 (Amphibia, Aves, Insecta, Mammalia, Reptilia)

class BirdModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=NUM_CLASSES, num_groups=NUM_GROUPS):
        super(BirdModel, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1)
        in_features = self.backbone.classifier.in_features

        # 移除原本的 classifier，改成只輸出特徵
        self.backbone.classifier = nn.Identity()

        # 主分類頭：物種
        self.fc_species = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

        # 輔助分類頭：大類（訓練用，推論時不使用）
        self.fc_class = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_groups)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits_species = self.fc_species(features)
        if self.training:
            logits_class = self.fc_class(features)
            return logits_species, logits_class
        return logits_species


if __name__ == '__main__':
    model = BirdModel()
    model.train()
    dummy = torch.randn(8, 1, 128, 313)
    out_species, out_class = model(dummy)
    print(f"訓練模式 - species: {out_species.shape}, class: {out_class.shape}")
    model.eval()
    out = model(dummy)
    print(f"推論模式 - species: {out.shape}")
