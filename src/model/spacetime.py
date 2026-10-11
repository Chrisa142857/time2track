import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./')

from einops import rearrange, repeat
from model.dit import *
from diffusers.models.embeddings import LabelEmbedding 

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=96, dim=512):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


@maybe_allow_in_graph
class SpatialTemporalTransformerBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        self.norm_temp = AdaLayerNorm(dim, chunk_dim=1)
        self.attn_temp = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        temb_in = temb
        text_seq_length = encoder_hidden_states.size(1)

        B, F, N, C = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, N, C)
        if encoder_hidden_states.shape[0] != B * F:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(F, 0)
        temb = temb_in.repeat_interleave(F, 0)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # Spatial Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        ## Time Attention
        hidden_states = rearrange(hidden_states, '(b f) n c -> (b n) f c', f=F)
        temb = temb_in.repeat_interleave(N, 0)
        norm_hidden_states = self.norm_temp(hidden_states, temb=temb)
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, '(b n) f c -> b f n c', n=N)

        # hidden_states = rearrange(hidden_states, '(b f) n c -> b f n c', f=F)

        return hidden_states, encoder_hidden_states

@maybe_allow_in_graph
class SpatialOnlyTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        temb_in = temb
        text_seq_length = encoder_hidden_states.size(1)

        B, F, N, C = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, N, C)
        if encoder_hidden_states.shape[0] != B * F:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(F, 0)
        temb = temb_in.repeat_interleave(F, 0)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # Spatial Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        hidden_states = rearrange(hidden_states, '(b f) n c -> b f n c', f=F)

        return hidden_states, encoder_hidden_states

@maybe_allow_in_graph
class TemporalOnlyTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        # self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # self.attn1 = Attention(
        #     query_dim=dim,
        #     dim_head=attention_head_dim,
        #     heads=num_attention_heads,
        #     qk_norm="layer_norm" if qk_norm else None,
        #     eps=1e-6,
        #     bias=attention_bias,
        #     out_bias=attention_out_bias,
        #     processor=CogVideoXAttnProcessor2_0(),
        # )

        self.norm_temp = AdaLayerNorm(dim, chunk_dim=1)
        self.attn_temp = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        temb_in = temb
        text_seq_length = encoder_hidden_states.size(1)

        B, F, N, C = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, N, C)
        if encoder_hidden_states.shape[0] != B * F:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(F, 0)
        temb = temb_in.repeat_interleave(F, 0)

        # # norm & modulate
        # norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
        #     hidden_states, encoder_hidden_states, temb
        # )

        # # Spatial Attention
        # attn_hidden_states, attn_encoder_hidden_states = self.attn1(
        #     hidden_states=norm_hidden_states,
        #     encoder_hidden_states=norm_encoder_hidden_states,
        #     image_rotary_emb=image_rotary_emb,
        # )

        # hidden_states = hidden_states + gate_msa * attn_hidden_states
        # encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        ## Time Attention
        hidden_states = rearrange(hidden_states, '(b f) n c -> (b n) f c', f=F)
        temb = temb_in.repeat_interleave(N, 0)
        norm_hidden_states = self.norm_temp(hidden_states, temb=temb)
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, '(b n) f c -> b f n c', n=N)

        # hidden_states = rearrange(hidden_states, '(b f) n c -> b f n c', f=F)

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class SpatialTemporalTransformerBlockv2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        self.norm_temp = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.attn_temp = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_time: torch.Tensor,
        temb: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        temb_in = temb
        text_seq_length = encoder_hidden_states.size(1)

        B, F, N, C = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, N, C)
        if encoder_hidden_states.shape[0] != B * F:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(F, 0)
        temb = temb_in.repeat_interleave(F, 0)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # Spatial Attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            indices=indices,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        ## Time Attention
        hidden_states = rearrange(hidden_states, '(b f) n c -> (b n) f c', f=F)
        temb = temb_in.repeat_interleave(N, 0)
        norm_hidden_states, norm_encoder_hidden_states_time, gate_msa, enc_gate_msa = self.norm_temp(
            hidden_states, encoder_hidden_states_time, temb
        )
        attn_hidden_states, attn_encoder_hidden_states_time = self.attn_temp(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states_time
        )
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states_time = encoder_hidden_states_time + enc_gate_msa * attn_encoder_hidden_states_time
        hidden_states = rearrange(hidden_states, '(b n) f c -> b f n c', n=N)

        return hidden_states, encoder_hidden_states, encoder_hidden_states_time

class SpaitalTemporalTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 3,
        out_channels: Optional[int] = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 8,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_points: int = 2048,
        sample_frames: int = 48,
        patch_size: int = 1,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        cond_seq_length: int = 4,
        cond_seq_length_t: int = 2,
        transformer_block: str = "SpatialTemporalTransformerBlock",
        num_classes: int = 0,
        class_dropout_prob: float = 0.0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if use_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        self.class_embedder = None
        if num_classes > 0:
            self.class_embedder = LabelEmbedding(num_classes, time_embed_dim, class_dropout_prob)

        self.transformer_block = transformer_block
        if transformer_block == "SpatialTemporalTransformerBlock":
            TransformerBlock = SpatialTemporalTransformerBlock
        elif transformer_block == "SpatialTemporalTransformerBlockv2":
            TransformerBlock = SpatialTemporalTransformerBlockv2
        elif transformer_block == "SpatialTemporalTransformerBlockv3":
            TransformerBlock = SpatialTemporalTransformerBlockv3
        elif transformer_block == "SpatialOnlyTransformerBlock":
            TransformerBlock = SpatialOnlyTransformerBlock
        elif transformer_block == "TemporalOnlyTransformerBlock":
            TransformerBlock = TemporalOnlyTransformerBlock

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False
        
        if use_positional_embeddings or use_learned_positional_embeddings:
            self.embed_dim = num_attention_heads * attention_head_dim
            self.cond_seq_length = cond_seq_length
            self.cond_seq_length_t = cond_seq_length_t
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_points, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(self, points: int, frames: int, device: Optional[torch.device] = None) -> torch.Tensor:
        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            points,
            frames,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)
        joint_pos_embedding = pos_embedding.new_zeros(
            1, self.cond_seq_length + points * frames, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data[:, self.cond_seq_length:].copy_(pos_embedding)
        return joint_pos_embedding
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def forward(
        self,
        hidden_states: torch.Tensor, # [batch_size]
        encoder_hidden_states: torch.Tensor, # [batch_size]
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        force_drop_ids: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        indices: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # TODO: check force drop id shape
        if self.class_embedder is not None:
            assert class_labels is not None
            class_labels = self.class_embedder(class_labels, force_drop_ids=force_drop_ids) # (N, D)
            emb = emb + class_labels

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb
        
        B, F, N, C = hidden_states.shape
        full_seq = torch.cat([encoder_hidden_states, hidden_states.reshape(B, F*N, -1)], axis=1)

        # 2. Patch embedding
        pos_embedding = self.pos_embedding
        pos_embedding = pos_embedding.to(dtype=full_seq.dtype)
        hidden_states = full_seq + pos_embedding
        
        hidden_states = self.embedding_dropout(hidden_states)
        encoder_hidden_states = hidden_states[:, :self.cond_seq_length]
        hidden_states = hidden_states[:, self.cond_seq_length:].reshape(B, F, N, C)

        if self.transformer_block not in ["SpatialTemporalTransformerBlock", 'TemporalOnlyTransformerBlock', 'SpatialOnlyTransformerBlock']:
            encoder_hidden_states_time = hidden_states[:, :self.cond_seq_length_t]
            encoder_hidden_states_time = rearrange(encoder_hidden_states_time, 'b f n c -> (b n) f c')
            hidden_states = hidden_states[:, self.cond_seq_length_t:]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if self.transformer_block in ["SpatialTemporalTransformerBlock", 'TemporalOnlyTransformerBlock', 'SpatialOnlyTransformerBlock']:
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states, encoder_hidden_states_time = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_time,
                        emb,
                        image_rotary_emb,
                        indices=indices,
                        **ckpt_kwargs,
                    )
            else:
                if self.transformer_block in ["SpatialTemporalTransformerBlock", 'TemporalOnlyTransformerBlock', 'SpatialOnlyTransformerBlock']:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )
                else:
                    hidden_states, encoder_hidden_states, encoder_hidden_states_time = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_time=encoder_hidden_states_time,
                        temb=emb,
                        indices=indices,
                        image_rotary_emb=image_rotary_emb,
                    )
        hidden_states = rearrange(hidden_states, 'b f n c -> b (f n) c')
        # 4. Final block
        hidden_states = self.norm_final(hidden_states)
        hidden_states = self.norm_out(hidden_states, temb=emb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        return output

class MDM_ST(nn.Module):
    def __init__(self, n_points, n_frame, n_feats, model_config):
        super().__init__()
        print('use new model')

        self.n_points = n_points
        self.n_feats = n_feats
        self.latent_dim = model_config.latent_dim
        self.cond_frame = 1 if model_config.frame_cond else 0
        self.frame_cond = model_config.frame_cond

        if model_config.get('point_embed', True):
            self.input_encoder = PointEmbed(dim=self.latent_dim)
        else:
            print('not using point embedding')
            self.input_encoder = nn.Linear(n_feats, self.latent_dim)
        self.mask_cond = model_config.get('mask_cond', False)
        if self.mask_cond:
            print('Use mask condition')
            self.mask_encoder = nn.Linear(1, self.latent_dim)
            self.cond_frame += 1
        self.pred_offset = model_config.get('pred_offset', True)
        self.num_neighbors = model_config.get('num_neighbors', 0)
        self.max_num_forces = model_config.get('max_num_forces', 1)
        self.model_config = model_config

        self.cond_seq_length = 2

        self.E_cond_encoder = nn.Linear(1, self.latent_dim)
        self.nu_cond_encoder = nn.Linear(1, self.latent_dim)
        self.force_as_token = model_config.get('force_as_token', True)
        self.force_as_latent = model_config.get('force_as_latent', False)

        if self.force_as_latent:
            self.input_encoder = nn.Linear(n_feats + 4 * self.max_num_forces, self.latent_dim)
        elif self.force_as_token:
            self.cond_seq_length += self.max_num_forces * 2
            self.force_cond_encoder = nn.Linear(3, self.latent_dim)
            self.drag_point_encoder = nn.Linear(3, self.latent_dim)
        else:
            self.cond_seq_length += 2
            self.force_cond_encoder = nn.Linear(3, self.latent_dim)
            self.drag_point_encoder = nn.Linear(3, self.latent_dim)

        self.gravity_emb = model_config.get('gravity_emb', False)
        if self.gravity_emb:
            self.gravity_embedding = nn.Embedding(2, self.latent_dim)
            self.cond_seq_length += 1

        if self.model_config.floor_cond:
            self.floor_encoder = nn.Linear(1, self.latent_dim)
            self.cond_seq_length += 1
        
        if self.model_config.coeff_cond:
            self.coeff_encoder = nn.Linear(1, self.latent_dim)
            self.cond_seq_length += 1

        self.num_mat = model_config.get('num_mat', 0)
        if model_config.class_token:
            self.class_embedding = nn.Embedding(model_config.num_mat, self.latent_dim)
            self.cond_seq_length += 1

        self.class_dropout_prob = model_config.get('class_dropout_prob', 0.0)
        self.dit = SpaitalTemporalTransformer(sample_points=n_points, sample_frames=n_frame+self.cond_frame, in_channels=n_feats,
            num_layers=model_config.n_layers, num_attention_heads=self.latent_dim // 64, time_embed_dim=self.latent_dim, cond_seq_length=self.cond_seq_length, cond_seq_length_t=self.cond_frame, transformer_block=model_config.transformer_block, num_classes=self.num_mat, class_dropout_prob=self.class_dropout_prob)
        
        self._init_weights()

    def _init_weights(self):
        if self.gravity_emb:
            nn.init.normal_(self.gravity_embedding.weight, mean=0.0, std=0.1)
    
    def enable_gradient_checkpointing(self):
        self.dit._set_gradient_checkpointing(True)

    def forward(self, x, timesteps, init_pc, force, E, nu, drag_mask, drag_point, floor_height, gravity_label=None, coeff=None, y=None, null_emb=None):
        
        """
        x: [batch_size, frame, n_points, n_feats], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        
        bs, n_frame, n_points, n_feats = x.shape
        
        init_pc = init_pc.reshape(bs, n_points, n_feats)
        force = force.unsqueeze(1) if force.ndim == 2 else force
        drag_point = drag_point.unsqueeze(1) if drag_point.ndim == 2 else drag_point
        E = E.unsqueeze(1)
        nu = nu.unsqueeze(1)

        if self.num_neighbors > 0:
            rel_dist = torch.cdist(init_pc, init_pc)
            dist, indices = rel_dist.topk(self.num_neighbors, largest = False)
            indices = indices.repeat_interleave(n_frame, 0)
            # indices = torch.cat([indices, torch.tensor([2048, 2049, 2050, 2051])[None, None].repeat(bs*n_frame, n_points, 1).to(indices.device)], axis=2)
        else:
            indices = None
        
        if self.force_as_token:
            force_emb = self.force_cond_encoder(force) + self.gravity_embedding(gravity_label) if self.gravity_emb else self.force_cond_encoder(force)
            encoder_hidden_states = torch.cat([self.E_cond_encoder(E), self.nu_cond_encoder(nu)], axis=1)
            # force_info = torch.cat([force, drag_point], dim=-1) # (B, n_forces, 7)
            # force_tokens = self.force_cond_encoder(force_info)
            encoder_hidden_states = torch.cat([encoder_hidden_states, force_emb, self.drag_point_encoder(drag_point[..., :3])], axis=1)
        elif self.force_as_latent:
            encoder_hidden_states = torch.cat([self.E_cond_encoder(E), self.nu_cond_encoder(nu)], axis=1)
            force = force.unsqueeze(1).repeat(1, n_points, 1, 1) # (B, n_points, n_forces, 3)
            all_force = torch.cat([force, drag_mask.permute(0, 2, 1, 3)], dim=-1).reshape(bs, n_points, -1) # (B, n_points, n_forces, 4)
        else:
            encoder_hidden_states = torch.cat([self.force_cond_encoder(force), self.E_cond_encoder(E),
                self.nu_cond_encoder(nu), self.drag_point_encoder(drag_point[..., :3])], axis=1) 
        if self.gravity_emb:
            encoder_hidden_states = torch.cat([encoder_hidden_states, self.gravity_embedding(gravity_label)], axis=1)
        if self.model_config.class_token:
            class_labels = y.unsqueeze(1)
            class_labels = self.class_embedding(class_labels)
            encoder_hidden_states = torch.cat([encoder_hidden_states, class_labels], axis=1)
        if self.model_config.floor_cond:
            floor_height = floor_height.unsqueeze(1) if floor_height is not None else None
            encoder_hidden_states = torch.cat([encoder_hidden_states, self.floor_encoder(floor_height)], axis=1)
        if self.model_config.coeff_cond:
            coeff = coeff.unsqueeze(1) if coeff is not None else None
            encoder_hidden_states = torch.cat([encoder_hidden_states, self.coeff_encoder(coeff)], axis=1)
        if null_emb is not None:
            encoder_hidden_states = encoder_hidden_states * null_emb
        if self.frame_cond:
            x = torch.cat([init_pc.unsqueeze(1), x], axis=1) # Condition on first frame
        if self.force_as_latent:
            all_force = all_force.unsqueeze(1).repeat(1, x.shape[1], 1, 1) # (B, n_frame, n_points, n_forces*4)
            x = torch.cat([x, all_force], dim=-1) # (B, n_frame, n_points, n_feats+n_forces * 4)
            n_feats = x.shape[-1]
        hidden_states = self.input_encoder(x.reshape(-1, n_points,
            n_feats)).reshape(bs, -1, n_points, self.latent_dim)
        if self.mask_cond:
            mask = self.mask_encoder(drag_mask[:, :1])
            hidden_states = torch.cat([mask, hidden_states], axis=1)
        if self.model_config.transformer_block in ["SpatialTemporalTransformerBlock", "TemporalOnlyTransformerBlock", "SpatialOnlyTransformerBlock"]:
            output = self.dit(hidden_states, encoder_hidden_states, timesteps, class_labels=y).reshape(bs, -1, n_points, 3)[:, self.cond_frame:]
        else:
            output = self.dit(hidden_states, encoder_hidden_states, timesteps, indices=indices).reshape(bs, -1, n_points, 3)
        output = output + init_pc.unsqueeze(1) if self.pred_offset else output
            
        return output

if __name__ == "__main__":

    # Diffusion
    from omegaconf import OmegaConf
    from options import TestingConfig
    cfg_path = '../traj-diff/configs/eval.yaml'
    config_path = 'model_config.yaml'
    device = 'cuda'
    schema = OmegaConf.structured(TestingConfig)
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(schema, cfg)
    n_training_frames = cfg.train_dataset.n_training_frames
    n_frames_interval = cfg.train_dataset.n_frames_interval

    point_num = 2048
    frame_num = 24
    x = torch.randn(1, frame_num, point_num, 3).to(device).to(torch.float16)
    timesteps = torch.tensor([999]).int().to(device).to(torch.float16)
    init_pc = torch.randn(1, 1, point_num, 3).to(device).to(torch.float16)
    force = torch.randn(1, 3).to(device).to(torch.float16)
    E = torch.randn(1, 1).to(device).to(torch.float16)
    nu = torch.randn(1, 1).to(device).to(torch.float16)

    x = nn.Parameter(x)

    with torch.enable_grad():
    # with torch.no_grad():
        t_total = 0 
        for i in range(100):
            model = MDM_ST(point_num, frame_num, 3, cfg.model_config).to(device).to(torch.float16)
            model.train()
            import time
            t0 = time.time()
            output = model(x, timesteps, init_pc, force, E, nu, None, force, torch.zeros_like(E), torch.ones_like(E), None)
            loss = output.sum()
            loss.backward()
            t1 = time.time()
            if i > 10:
                t_total += t1 - t0
            print(t1 - t0)

    print("Average time: ", t_total / 90)