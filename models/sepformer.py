
# copy from https://github.com/lin9x/AV-Sepformer/blob/master/models/sepformer.py and simple_modifications

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
import torch
import torch.nn as nn

class SI_SDR(nn.Module):
    def __init__(self):
        super(SI_SDR, self).__init__()
        self.EPS = 1e-8
    def forward(self, reference, estimate):
        # 确保输入长度一致
        if reference.shape[-1] > estimate.shape[-1]:
            reference = reference[..., :estimate.shape[-1]]
        elif reference.shape[-1] < estimate.shape[-1]:
            estimate = estimate[..., :reference.shape[-1]]

        # 计算信号能量
        ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + self.EPS
        proj = torch.sum(reference * estimate, dim=-1, keepdim=True) * reference / ref_energy

        # 计算噪声
        noise = estimate - proj
        # SI-SDR
        si_sdr = 10 * torch.log10(torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + self.EPS))

        return -torch.mean(si_sdr)  # 转化为损失

def bss_sdr(target, estimate,eps=1e-8):
    """
    与mir_eval的bss_eval_sources完全对齐的SDR计算函数
    
    参数:
        estimate (Tensor): 估计信号，(batch, samples) 或 (samples,)
        target (Tensor): 目标信号，与estimate同维度
        eps (float): 数值稳定项
        
    返回:
        Tensor: SDR值(dB)，shape: (batch,)
    """
    # 维度处理
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    # 零均值处理 (与bss_eval_sources一致)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # 计算投影系数 (最小二乘最优缩放因子)
    dot = torch.sum(target * estimate, dim=-1, keepdim=True)  # (batch,1)
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + eps  # (batch,1)
    alpha = dot / target_energy  # (batch,1)

    # 分解信号分量
    e_target = alpha * target  # 投影到目标方向
    e_res = estimate - e_target  # 残差分量

    # 计算功率比
    target_power = torch.sum(e_target**2, dim=-1)
    res_power = torch.sum(e_res**2, dim=-1)
    sdr = 10 * torch.log10((target_power + eps) / (res_power + eps))

    return -sdr
class SI_SNR(nn.Module):
    def __init__(self):
        super(SI_SNR, self).__init__()
        self.EPS = 1e-8

    def forward(self, source, estimate_source):
        if source.shape[-1] > estimate_source.shape[-1]:
            source = source[..., :estimate_source.shape[-1]]
        if source.shape[-1] < estimate_source.shape[-1]:
            estimate_source = estimate_source[..., :source.shape[-1]]

        # step 1: Zero-mean norm
        source = source - torch.mean(source, dim=-1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

        # step 2: Cal si_snr
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, dim = -1, keepdim=True) + self.EPS
        proj = torch.sum(source * estimate_source, dim = -1, keepdim=True) * source / ref_energy
        # e_noise = s' - s_target
        noise = estimate_source - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, dim = -1) / (torch.sum(noise ** 2, dim = -1) + self.EPS)
        sisnr = 10 * torch.log10(ratio + self.EPS)

        return 0 - torch.mean(sisnr)

class MuSE_loss(nn.Module):
    def __init__(self):
        super(MuSE_loss, self).__init__()
        self.si_snr_loss = SI_SNR()
        self.speaker_loss = nn.CrossEntropyLoss()

    def forward(self, tgt_wav, pred_wav, tgt_spk, pred_spk):
        si_snr = self.si_snr_loss(tgt_wav, pred_wav)
        ce = self.speaker_loss(pred_spk[0], tgt_spk) + self.speaker_loss(pred_spk[1], tgt_spk) + self.speaker_loss(pred_spk[2], tgt_spk) + self.speaker_loss(pred_spk[3], tgt_spk)
        return {'si_snr': si_snr, 'ce': ce} 

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #self.register_buffer('pe', pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_parameter("inv_freq", nn.Parameter(inv_freq, requires_grad=False))
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) +
                     self.bias)
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm,
              self).__init__(dim,
                             elementwise_affine=elementwise_affine,
                             eps=1e-8)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """
    def __init__(self, kernel_size=16, out_channels=256, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(768)
        dsconv = nn.Conv1d(768,
                           768,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=768,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(768)
        pw_conv = nn.Conv1d(768, 768, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x


class cross_attention_layer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, norm_first=True, *args, **kwargs):
        super().__init__(d_model,
                         nhead,
                         norm_first=norm_first,
                         *args,
                         **kwargs)

    def forward(self, x, v):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        if self.norm_first:
            x = x + v + self._ca_block(self.norm1(x), self.norm1(v))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._ca_block(x, v))
            x = self.norm2(x + self._ff_block(v))

        return x

    def _ca_block(self, x, v):
        x = self.self_attn(v,
                           x,
                           x,
                           attn_mask=None,
                           key_padding_mask=None,
                           need_weights=False)[0]
        return self.dropout1(x)


class CrossTransformer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 depth=4,
                 dropout=0.1,
                 dim_feedforward=2048,
                 activation=F.relu):
        super(CrossTransformer, self).__init__()
        self.cross_attention_layer = cross_attention_layer(
            d_model,
            nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
            norm_first=True)
        self.transformer_layers = []
        for _ in range(depth - 1):
            self.transformer_layers.append(
                nn.TransformerEncoderLayer(d_model,
                                           nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout,
                                           activation=activation,
                                           norm_first=True))
        self.transformer_layers = nn.Sequential(*self.transformer_layers)

    def forward(self, video, audio):
        x = self.cross_attention_layer(audio, video)
        x = self.transformer_layers(x)
        return x


class CrossTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        dropout=0.0,
        activation="relu",
        use_positional_encoding=True,
        norm_before=True,
    ):
        super(CrossTransformerBlock, self).__init__()

        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = CrossTransformer(d_model,
                                    nhead,
                                    dim_feedforward=d_ffn,
                                    depth=num_layers,
                                    dropout=dropout,
                                    activation=activation)

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model, dropout=0.0)

    def forward(self, x, video):
        x = x.permute(1, 0, 2)
        video = video.permute(1, 0, 2)
        if self.use_positional_encoding:
            x = self.pos_enc(x)
            video = self.pos_enc(video)
            x = self.mdl(video, x)
        else:
            x = self.mdl(video, x)
        return x.permute(1, 0, 2)


class SBTransformerBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        dropout=0.0,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
            activation = F.relu
        elif activation == "gelu":
            activation = F.gelu
        else:
            raise ValueError("unknown activation")
        self.mdl = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=nhead,
                                                     dim_feedforward=d_ffn,
                                                     dropout=dropout,
                                                     activation=activation,
                                                     norm_first=norm_before),
            num_layers=num_layers)

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        x = x.permute(1, 0, 2)
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            x = self.mdl(pos_enc)
        else:
            x = self.mdl(x)
        return x.permute(1, 0, 2)


class Cross_Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """
    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Cross_Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        #self.pos2d = PositionalEncodingPermute2D(256)
        self.pos2d = PositionalEncodingPermute2D(out_channels)

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

    def forward(self, x, v):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        pe = self.pos2d(x)
        x = x + pe
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        B_v, N_v, S_v = v.shape
        v = v + pe[:, :, K // 2, :]
        v = v.unsqueeze(-2).repeat(1, 1, K, 1)
        v = v.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)

        inter = self.inter_mdl(inter, v)

        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class Cross_Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=160,
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Cross_Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)
        ve_blocks = []
        for _ in range(5):
            ve_blocks += [VisualConv1D()]
        ve_blocks += [nn.Conv1d(768, out_channels, 1)]
        self.visual_conv = nn.Sequential(*ve_blocks)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Cross_Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                    )))

        self.conv2d = nn.Conv2d(out_channels,
                                out_channels * num_spks,
                                kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x, video):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        video = video.transpose(1, 2)
        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(
                1, -1) + x * (x.size(1)**0.5)

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        v = self.visual_conv(video)
        v = F.pad(v, (0, x.shape[-1] - v.shape[-1]), mode='replicate')
        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x, v)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (torch.cat([input1, input2], dim=3).view(B, N, -1,
                                                         K).transpose(2, 3))

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Cross_Sepformer(nn.Module):
    def __init__(self,
                 IntraSeparator,
                 InterSeparator,
                 kernel_size=16,
                 N_encoder_out=256,
                 num_spks=1):
        super(Cross_Sepformer, self).__init__()

        self.AudioEncoder = Encoder(kernel_size=kernel_size,
                                    out_channels=N_encoder_out)

        self.AudioDecoder = Decoder(in_channels=N_encoder_out,
                                    out_channels=1,
                                    kernel_size=kernel_size,
                                    stride=kernel_size // 2,
                                    bias=False)
        self.Separator = Cross_Dual_Path_Model(num_spks=num_spks,
                                               in_channels=N_encoder_out,
                                               out_channels=N_encoder_out,
                                               num_layers=2,
                                               K=160,
                                               intra_model=IntraSeparator,
                                               inter_model=InterSeparator,
                                               norm='ln',
                                               skip_around_intra=True)
        self.num_spks = num_spks

    def forward(self, mix, video):
        mix_w = self.AudioEncoder(mix)
        est_mask = self.Separator(mix_w, video)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.AudioDecoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source.permute(0, 2, 1).squeeze(1)


def build_Sepformer(kernel_size=16, N_encoder_out=384, num_spks=1,num_layers=12):
    InterSeparator = CrossTransformerBlock(num_layers=num_layers,
                                           d_model=N_encoder_out,
                                           nhead=8,
                                           d_ffn=1024,
                                           dropout=0,
                                           use_positional_encoding=False,
                                           norm_before=True)
    IntraSeparator = SBTransformerBlock(num_layers=num_layers,
                                        d_model=N_encoder_out,
                                        nhead=8,
                                        d_ffn=1024,
                                        dropout=0,
                                        use_positional_encoding=True,
                                        norm_before=True)
    return Cross_Sepformer(IntraSeparator,
                           InterSeparator,
                           kernel_size=kernel_size,
                           N_encoder_out=N_encoder_out,
                           num_spks=num_spks)
