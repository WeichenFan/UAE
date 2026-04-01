# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def create_dct_matrix(n, device=None, dtype=torch.float32):
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)

    mat = torch.cos(math.pi / n * (i + 0.5) * k)
    mat[0] = mat[0] * math.sqrt(1.0 / n)
    mat[1:] = mat[1:] * math.sqrt(2.0 / n)
    return mat


def dct_2d(x, dct_mat):
    x = torch.matmul(dct_mat, x)
    x = torch.matmul(x, dct_mat.transpose(-1, -2))
    return x


def idct_2d(x, dct_mat):
    x = torch.matmul(dct_mat.transpose(-1, -2), x)
    x = torch.matmul(x, dct_mat)
    return x


def window_partition(x, window_size):
    b, c, h, w = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(
            f"H and W must be divisible by window_size={window_size}, got H={h}, W={w}"
        )

    x = x.reshape(b, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, (h // window_size) * (w // window_size), c, window_size, window_size)


def window_reverse(windows, window_size, h, w):
    b, num_windows, c, p1, p2 = windows.shape
    if p1 != window_size or p2 != window_size:
        raise ValueError(
            f"Window tensor shape {tuple(windows.shape)} does not match window_size={window_size}"
        )
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError(
            f"H and W must be divisible by window_size={window_size}, got H={h}, W={w}"
        )

    h_blocks = h // window_size
    w_blocks = w // window_size
    if num_windows != h_blocks * w_blocks:
        raise ValueError(
            f"Expected {h_blocks * w_blocks} windows for shape ({h}, {w}), got {num_windows}"
        )

    x = windows.reshape(b, h_blocks, w_blocks, c, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(b, c, h, w)


def build_frequency_radius(window_size, device=None, dtype=torch.float32):
    u = torch.arange(window_size, device=device, dtype=dtype)
    v = torch.arange(window_size, device=device, dtype=dtype)
    uu, vv = torch.meshgrid(u, v, indexing="ij")

    radius = torch.sqrt(uu ** 2 + vv ** 2)
    max_radius = math.sqrt((window_size - 1) ** 2 + (window_size - 1) ** 2)
    return radius / max_radius


def build_soft_radial_mask(
    window_size,
    scale,
    sharpness=30.0,
    min_keep=0.05,
    max_keep=1.0,
    device=None,
    dtype=torch.float32,
):
    if scale.dim() == 0:
        scale = scale.unsqueeze(0)

    scale = scale.clamp(0.0, 1.0)
    cutoff = min_keep + (max_keep - min_keep) * scale

    radius = build_frequency_radius(window_size, device=device, dtype=dtype).unsqueeze(0)
    cutoff = cutoff[:, None, None]
    mask = torch.sigmoid(sharpness * (cutoff - radius))
    return mask[:, None, None, :, :]


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        window_size=None,
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True,
        mask_sharpness=30.0,
        min_keep=0.05,
        max_keep=1.0,
        use_reconstruction=True,
    ):
        super().__init__()
        if window_size is None:
            window_size = img_size
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        window_size = (window_size, window_size)
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Only square patch sizes are supported, got {patch_size}.")
        if window_size[0] != window_size[1]:
            raise ValueError(f"Only square window sizes are supported, got {window_size}.")
        if img_size[0] % patch_size[0] != 0 or img_size[1] % patch_size[1] != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}.")
        if img_size[0] % window_size[0] != 0 or img_size[1] % window_size[1] != 0:
            raise ValueError(f"img_size={img_size} must be divisible by window_size={window_size}.")
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.pca_dim = pca_dim
        self.mask_sharpness = mask_sharpness
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.use_reconstruction = use_reconstruction

        patch_dim = in_chans * patch_size[0] * patch_size[1]
        self.proj1 = nn.Linear(patch_dim, pca_dim, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.register_buffer(
            "dct_mat",
            create_dct_matrix(window_size[0], dtype=torch.float32),
            persistent=False
        )

    def _normalize_scale(self, scale, batch_size, device, dtype):
        if scale is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, device=device, dtype=dtype)
        scale = scale.to(device=device, dtype=dtype)
        if scale.dim() == 0:
            scale = scale.expand(batch_size)
        elif scale.dim() == 1 and scale.shape[0] == batch_size:
            pass
        elif scale.shape[0] == batch_size:
            scale = scale.reshape(batch_size, -1)[:, 0]
        else:
            raise ValueError(f"scale must be scalar or batch-aligned, got shape {tuple(scale.shape)}")
        return scale.clamp(0.0, 1.0)

    def forward(self, x, scale=None):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if C != self.in_chans:
            raise ValueError(f"Expected in_chans={self.in_chans}, got {C}")

        patch_size = self.patch_size[0]
        window_size = self.window_size[0]
        scale = self._normalize_scale(scale, B, x.device, x.dtype)

        windows = window_partition(x, window_size)
        _, num_windows, _, _, _ = windows.shape

        dct_mat = self.dct_mat.to(device=x.device, dtype=x.dtype)
        freq = dct_2d(windows.reshape(B * num_windows * C, window_size, window_size), dct_mat)
        freq = freq.reshape(B, num_windows, C, window_size, window_size)

        mask = build_soft_radial_mask(
            window_size=window_size,
            scale=scale,
            sharpness=self.mask_sharpness,
            min_keep=self.min_keep,
            max_keep=self.max_keep,
            device=x.device,
            dtype=x.dtype,
        )
        masked_freq = freq * mask

        if self.use_reconstruction:
            reconstructed_windows = idct_2d(
                masked_freq.reshape(B * num_windows * C, window_size, window_size),
                dct_mat,
            ).reshape(B, num_windows, C, window_size, window_size)
            x = window_reverse(reconstructed_windows, window_size, H, W)
            token_source = window_partition(x, patch_size)
        else:
            if window_size != patch_size:
                raise ValueError(
                    "window_size must equal patch_size when use_reconstruction is False."
                )
            token_source = masked_freq

        num_tokens = token_source.shape[1]
        x = token_source.reshape(B, num_tokens, C * patch_size * patch_size)
        x = self.proj1(x)
        x = x.transpose(1, 2).reshape(B, self.pca_dim, H // patch_size, W // patch_size)
        x = self.proj2(x).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        window_size=None,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.window_size = patch_size if window_size is None else window_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            self.window_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
            use_reconstruction=True,
        )

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len
        )

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1)
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        if self.x_embedder.proj2.bias is not None:
            nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """
        # class and time embeddings
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        # forward JiT
        x = self.x_embedder(x, scale=t)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self.feat_rope if i < self.in_context_start else self.feat_rope_incontext)

        x = x[:, self.in_context_len:]

        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        return output

def JiT_700M(**kwargs):
    return JiT(depth=36, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=12, patch_size=32, **kwargs)

def JiT_2B(**kwargs):
    return JiT(depth=27, hidden_size=2048, num_heads=32,
               bottleneck_dim=256, in_context_len=32, in_context_start=9, patch_size=32, **kwargs)

def JiT_7B(**kwargs):
    return JiT(depth=24, hidden_size=4096, num_heads=64,
               bottleneck_dim=512, in_context_len=32, in_context_start=8, patch_size=32, **kwargs)


JiT_models = {
    'JiT-700M': JiT_700M,
    'JiT-2B': JiT_2B,
    'JiT-7B': JiT_7B,
}
