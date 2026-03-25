import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# === 小工具：用來初始化卷積層，讓神經網路有一個好的起跑點 ===
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

# === PANNs 的標準建材：由兩層 3x3 卷積加上池化組成的 Block ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x

# === CNN10 模型本體 ===
class PANNsCNN10(nn.Module):
    def __init__(self, sample_rate=32000, window_size=512, hop_size=320, mel_bins=160, classes_num=234, num_groups=5):
        super(PANNsCNN10, self).__init__()

        # GPU 內建的超狂頻譜轉換器！直接掛在網路的最前端
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=window_size, 
            hop_length=hop_size, 
            n_mels=mel_bins, 
            f_min=50, 
            f_max=16000
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # 標準的 CNN10 四層過濾器
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # 最終層：全連接層輸出我們專案規定的 234 種鳥類
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        # 輔助分類頭：大類（訓練用，推論時不使用）
        self.fc_class = nn.Linear(512, num_groups, bias=True)

        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        init_layer(self.fc_class)

    def forward(self, input):
        # 1. 這裡的 input 是最原始的聲波 (Waveform)！例如 5 秒等於 160000 個採樣點
        x = self.mel_spectrogram(input)
        x = self.amplitude_to_db(x)

        # mel_spectrogram 輸出 (B, F, T)
        # bn0 = BatchNorm2d(mel_bins)，需要 F 在 dim=1：(B, F, 1, T)
        # 卷積期望 (B, 1, T, F)，BN 後再 transpose 回去
        x = x.unsqueeze(2)          # (B, F, 1, T)
        x = self.bn0(x)             # BN 作用在 F=mel_bins 維度
        x = x.squeeze(2)            # (B, F, T)
        x = x.transpose(1, 2)       # (B, T, F)
        x = x.unsqueeze(1)          # (B, 1, T, F)

        # 3. 經過四層高強度卷積，榨出鳥的形狀與特徵
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        x = self.conv_block4(x, pool_size=(2, 2))
        
        # 4. 全局平均池化 (把剩下的長與寬都濃縮，變成只有 512 個點的特徵條)
        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        
        # 5. 輸出預測
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 最終分數
        logits_species = self.fc_audioset(x)

        if self.training:
            logits_class = self.fc_class(x)
            return logits_species, logits_class
        return logits_species

if __name__ == '__main__':
    print("正在建構 CNN10 神經網路中...")
    model = PANNsCNN10()
    
    # 【火力展示】模擬送進去 8 個 Batch，每個都是剛好 5 秒鐘長度的原聲波！
    # (取樣率 32000 * 5 秒 = 160000)
    dummy_audio_waveform = torch.randn(8, 160000)
    
    output = model(dummy_audio_waveform)
    
    print("\n📦 送入的一車原始聲波長度為: ", dummy_audio_waveform.shape)
    print("👉 CNN10 分析出的鳥類預測結果為: ", output.shape)
