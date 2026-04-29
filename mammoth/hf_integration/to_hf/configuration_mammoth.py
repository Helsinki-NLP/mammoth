"""MammothConfig: PretrainedConfig for Mammoth x-transformers encoder-decoder models."""

from typing import Optional
from transformers import PretrainedConfig


class MammothConfig(PretrainedConfig):
    model_type = "mammoth"

    def __init__(
        self,
        # Vocabulary
        src_vocab_size: int = 50265,
        tgt_vocab_size: int = 50265,
        # Dimensions
        enc_model_dim: int = 768,
        dec_model_dim: int = 768,
        enc_layers: int = 6,
        dec_layers: int = 6,
        enc_max_seq_len: int = 1024,
        dec_max_seq_len: int = 1024,
        # Encoder attention
        enc_heads: int = 12,
        enc_attn_dim_head: int = 64,
        enc_attn_dropout: float = 0.0,
        enc_attn_qkv_bias: bool = True,
        enc_attn_flash: bool = False,
        # Encoder FFN
        enc_ff_mult: float = 4.0,
        enc_ff_glu: bool = False,
        enc_ff_no_bias: bool = False,
        enc_ff_dropout: float = 0.0,
        # Encoder norm / positional
        enc_pre_norm: bool = False,        # False = post-norm (BART style)
        enc_use_rmsnorm: bool = False,
        enc_layernorm_bias: bool = True,
        enc_norm_add_unit_offset: bool = False,
        enc_rotary_pos_emb: bool = False,
        enc_use_abs_pos_emb: bool = True,
        enc_post_emb_norm: bool = True,
        enc_post_emb_norm_bias: bool = True,
        enc_scaled_embeddings: bool = False,
        enc_emb_dropout: float = 0.0,
        enc_post_emb_norm_module: bool = False,  # unused in forward, kept for completeness
        # Encoder sliding window (Gemma3-style; -1 = full attention)
        enc_sliding_window: int = -1,
        # Decoder attention
        dec_heads: int = 12,
        dec_attn_dim_head: int = 64,
        dec_attn_dropout: float = 0.0,
        dec_attn_qkv_bias: bool = True,
        dec_attn_flash: bool = False,
        dec_attn_kv_heads: Optional[int] = None,     # None = multi-head (no MQA/GQA)
        dec_attn_qk_norm: bool = False,
        dec_attn_qk_norm_dim_scale: bool = False,
        dec_cross_attn_dim_context: Optional[int] = None,  # must equal enc_model_dim when set
        # Decoder FFN
        dec_ff_mult: float = 4.0,
        dec_ff_glu: bool = False,
        dec_ff_no_bias: bool = False,
        dec_ff_dropout: float = 0.0,
        # Decoder norm / positional
        dec_pre_norm: bool = False,
        dec_use_rmsnorm: bool = False,
        dec_layernorm_bias: bool = True,
        dec_norm_add_unit_offset: bool = False,
        dec_rotary_pos_emb: bool = False,
        dec_use_abs_pos_emb: bool = True,
        dec_post_emb_norm: bool = True,
        dec_post_emb_norm_bias: bool = True,
        dec_scaled_embeddings: bool = False,
        dec_emb_dropout: float = 0.0,
        dec_sandwich_norm: bool = False,
        dec_post_emb_norm_module: bool = False,
        # Decoder sliding window (Gemma3-style)
        dec_sliding_window: int = -1,
        dec_global_attn_every_n_layers: int = 0,
        dec_global_rope_theta: float = 10000.0,
        dec_local_rope_theta: float = 10000.0,
        # Shared options
        tie_word_embeddings: bool = True,
        model_dtype: str = "bf16",
        # Special token ids (passed through to PretrainedConfig)
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        decoder_start_token_id: int = 2,
        # KV cache: not implemented in this thin x-transformers wrapper. Disable
        # so HF's GenerationMixin does not allocate a DynamicCache (which would
        # read decoder_config.num_hidden_layers and crash).
        use_cache: bool = False,
        **kwargs,
    ):
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=True,
            use_cache=use_cache,
            **kwargs,
        )
        # Aliases some HF utilities probe on configs (e.g. cache layer counts).
        self.num_hidden_layers = dec_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.vocab_size = tgt_vocab_size  # required by HF GenerationMixin
        self.enc_model_dim = enc_model_dim
        self.dec_model_dim = dec_model_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len

        self.enc_heads = enc_heads
        self.enc_attn_dim_head = enc_attn_dim_head
        self.enc_attn_dropout = enc_attn_dropout
        self.enc_attn_qkv_bias = enc_attn_qkv_bias
        self.enc_attn_flash = enc_attn_flash
        self.enc_ff_mult = enc_ff_mult
        self.enc_ff_glu = enc_ff_glu
        self.enc_ff_no_bias = enc_ff_no_bias
        self.enc_ff_dropout = enc_ff_dropout
        self.enc_pre_norm = enc_pre_norm
        self.enc_use_rmsnorm = enc_use_rmsnorm
        self.enc_layernorm_bias = enc_layernorm_bias
        self.enc_norm_add_unit_offset = enc_norm_add_unit_offset
        self.enc_rotary_pos_emb = enc_rotary_pos_emb
        self.enc_use_abs_pos_emb = enc_use_abs_pos_emb
        self.enc_post_emb_norm = enc_post_emb_norm
        self.enc_post_emb_norm_bias = enc_post_emb_norm_bias
        self.enc_scaled_embeddings = enc_scaled_embeddings
        self.enc_emb_dropout = enc_emb_dropout
        self.enc_sliding_window = enc_sliding_window

        self.dec_heads = dec_heads
        self.dec_attn_dim_head = dec_attn_dim_head
        self.dec_attn_dropout = dec_attn_dropout
        self.dec_attn_qkv_bias = dec_attn_qkv_bias
        self.dec_attn_flash = dec_attn_flash
        self.dec_attn_kv_heads = dec_attn_kv_heads
        self.dec_attn_qk_norm = dec_attn_qk_norm
        self.dec_attn_qk_norm_dim_scale = dec_attn_qk_norm_dim_scale
        self.dec_cross_attn_dim_context = dec_cross_attn_dim_context
        self.dec_ff_mult = dec_ff_mult
        self.dec_ff_glu = dec_ff_glu
        self.dec_ff_no_bias = dec_ff_no_bias
        self.dec_ff_dropout = dec_ff_dropout
        self.dec_pre_norm = dec_pre_norm
        self.dec_use_rmsnorm = dec_use_rmsnorm
        self.dec_layernorm_bias = dec_layernorm_bias
        self.dec_norm_add_unit_offset = dec_norm_add_unit_offset
        self.dec_rotary_pos_emb = dec_rotary_pos_emb
        self.dec_use_abs_pos_emb = dec_use_abs_pos_emb
        self.dec_post_emb_norm = dec_post_emb_norm
        self.dec_post_emb_norm_bias = dec_post_emb_norm_bias
        self.dec_scaled_embeddings = dec_scaled_embeddings
        self.dec_emb_dropout = dec_emb_dropout
        self.dec_sandwich_norm = dec_sandwich_norm
        self.dec_sliding_window = dec_sliding_window
        self.dec_global_attn_every_n_layers = dec_global_attn_every_n_layers
        self.dec_global_rope_theta = dec_global_rope_theta
        self.dec_local_rope_theta = dec_local_rope_theta

        self.model_dtype = model_dtype