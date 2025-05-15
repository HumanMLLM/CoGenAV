import os
import sys
sys.path.append("../fairseq/")
import torch
from torch import nn
import omegaconf
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from fairseq.modules import LayerNorm
from .resnet import ResEncoder

class AudioFronted(nn.Module):
    def __init__(self, num_mel_bins=80, embed_dim=768):
        super(AudioFronted, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv1d(num_mel_bins, embed_dim//2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
    def forward(self, input_features):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        return inputs_embeds

class GatedFFN(nn.Module):
    """ 带有多重门控机制的前馈网络 """
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )
        self.value = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )
        
    def forward(self, x):
        return x + self.gate(x) * self.value(x)  # 残差门控
        
class FeatureAdaptation(nn.Module):
    def __init__(self, embed_dim=768,num_heads=8, num_layers=3, ffn_expansion=4,use_gate_ffn=True):
        super().__init__()
        if use_gate_ffn:
            ffn_layer = GatedFFN(embed_dim,ffn_expansion)  
        else:
            ffn_layer =nn.Sequential(                               # 前馈网络
                    nn.Linear(embed_dim, embed_dim*ffn_expansion),
                    nn.GELU(),
                    nn.Linear(embed_dim*ffn_expansion, embed_dim)
                )
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim, num_heads),  # 交叉注意力层
                "attn_norm": nn.LayerNorm(embed_dim),                # 注意力子层归一化
                "ffn":ffn_layer,
                "ffn_norm": nn.LayerNorm(embed_dim)                 # FFN子层归一化
            }) for _ in range(num_layers)
        ])
    def forward(self, audio_feature, visual_feature):
        """
        参数:
        audio_feature: 音频特征 [batch_size, 2T, 768]
        visual_feature: 视觉特征 [batch_size, T, 768] or [batch_size, 2T, 768]
        返回:
        增强后的音频特征 [batch_size, 2T, 768]
        """
        # 调整维度顺序以适应PyTorch多头注意力输入格式
        audio_seq = audio_feature.permute(1, 0, 2)  # [2T, batch_size, 768]
        visual_seq = visual_feature.permute(1, 0, 2)  # [T, batch_size, 768]
        # 逐层处理
        for layer in self.layers:
            # ========== 注意力子层 ==========
            # 交叉注意力计算
            attn_output, _ = layer["attn"](
                query=audio_seq,
                key=visual_seq,
                value=visual_seq,
                need_weights=False
            )
            # 残差连接 + 层归一化
            audio_seq = layer["attn_norm"](attn_output + audio_seq)
            
            # ========== FFN子层 ==========
            ffn_output = layer["ffn"](audio_seq)          # 前馈网络
            audio_seq = layer["ffn_norm"](ffn_output + audio_seq)  # 残差连接+归一化
        return audio_seq.permute(1, 0, 2)  # [batch_size, 2T, 768]

class DeltaUpsampler(nn.Module):
    """时序上采样模块"""
    def __init__(self, embed_dim):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Conv1d(embed_dim, 2*embed_dim, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, D, T]
        delta = self.delta_net(x)      # [B, 2D, T]
        even = x + delta[:, ::2]  # 偶数位置
        odd = x + delta[:, 1::2]  # 奇数位置
        return torch.stack([even, odd], dim=2).flatten(2).permute(0, 2, 1)

class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, encoder_embed_dim=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, encoder_embed_dim)
    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

class CoGenAVEncoder(nn.Module):
    #基于https://github.com/facebookresearch/av_hubert/blob/main/avhubert/hubert.py进行修改得到
    def __init__(self, cfg):
        super(CoGenAVEncoder, self).__init__()
        
        resnet = ResEncoder(relu_type='prelu', weights=None)
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.embed = 2 * self.encoder_embed_dim
        
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, encoder_embed_dim=self.encoder_embed_dim)
        self.feature_extractor_audio = AudioFronted(embed_dim=self.encoder_embed_dim)
        
        self.post_extract_proj = nn.Linear(self.embed, self.encoder_embed_dim)
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        self.dropout_input = nn.Dropout(cfg.dropout_input)

    def forward(self, video=None, audio=None):
        assert video is not None or audio is not None, "At least one of video or audio must be provided"
        if video is not None:
            features_video = self.feature_extractor_video(video)
        if audio is not None:
            features_audio = self.feature_extractor_audio(audio) # features: [B, F, T]
        if audio is None and video is not None:
            features_audio = 0 * features_video
        if video is None and audio is not None:
            features_video = 0 * features_audio

        features = torch.cat([features_audio, features_video], dim=1).transpose(1, 2)
        features = self.layer_norm(features)
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        x = self.dropout_input(features)
        x, _ = self.encoder(x, padding_mask=None, layer=None)
        
        return x

class CoGenAV(nn.Module):
    def __init__(self, cfg_file='large.yaml',model_tensor=None):
        super(CoGenAV, self).__init__()  # 调用父类的 __init__ 方法
        cfg = omegaconf.OmegaConf.load(cfg_file)

        self.encoder = CoGenAVEncoder(cfg)
        self.upsampler = DeltaUpsampler(embed_dim=cfg.encoder_embed_dim )
        self.adapter = FeatureAdaptation(embed_dim=cfg.encoder_embed_dim )

        if model_tensor is not None:
            resave_name =  f'weights/{os.path.basename(cfg_file)[:-5]}_cogen.pt'
            self._load_model_weights(model_tensor,resave_name)

    def _load_model_weights(self, model_tensor,resave_name=None):

        if model_tensor.endswith('_cogen.pt'):
            selected_tensors = torch.load(model_tensor, map_location='cpu')
        self.load_state_dict(selected_tensors,strict=True)

    def forward(self, video, audio=None,use_upsampler=True):
        """
        Forward method to process video data.
        Args:
            video_data (torch.Tensor): The input video data.
        Returns:
            dict: Output containing features and other information if applicable.
        """
        if video is not None:
            video = video.permute(0, 3, 4, 1, 2).contiguous()

        x = self.encoder(video, audio)

        if self.upsampler is not None and use_upsampler:
            x = self.upsampler(x)
        return x

def seq2seq_contrastive_loss(audio, video, y):
    """
    Calculate the seq2seq contrastive loss using functional API.
    Parameters:
    - audio: Tensor, the first input (audio features).
    - video: Tensor, the second input (video features).
    - y: Tensor, the target labels (should be 0 or 1).
    Returns:
    - loss: Tensor, the computed loss.
    """
    d = F.relu(F.cosine_similarity(audio, video, dim=2)).mean(-1)
    with torch.autocast(device_type=d.device.type, enabled=False):
        loss = F.binary_cross_entropy(d.unsqueeze(1), y.float())
    return loss


