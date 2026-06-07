"""MammothConfig: PretrainedConfig for native-PyTorch Mammoth encoder-decoder models."""

from typing import List, Optional
from transformers import PretrainedConfig


class MammothConfig(PretrainedConfig):
    model_type = "mammoth"

    def __init__(
        self,
        model_dim: int = 512,
        heads: int = 8,
        ff_mult: float = 2.67,
        ff_swiglu: bool = True,
        enc_layers: List[int] = (6,),
        dec_layers: List[int] = (6,),
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        rotary_pos_emb: bool = True,
        post_emb_norm: bool = True,
        src_vocab_size: int = 32000,
        tgt_vocab_size: int = 32000,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        decoder_start_token_id: int = 2,
        src_tokenizer_dir: str = "src_tokenizer",
        tgt_tokenizer_dir: str = "tgt_tokenizer",
        encoder_sharing_groups: Optional[List[str]] = None,
        decoder_sharing_groups: Optional[List[str]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=True,
            use_cache=use_cache,
            **kwargs,
        )
        self.model_dim = model_dim
        self.heads = heads
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu
        self.enc_layers = list(enc_layers)
        self.dec_layers = list(dec_layers)
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.emb_dropout = emb_dropout
        self.rotary_pos_emb = rotary_pos_emb
        self.post_emb_norm = post_emb_norm
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.vocab_size = tgt_vocab_size      # required by HF GenerationMixin
        self.num_hidden_layers = sum(dec_layers)  # HF cache utilities probe this
        self.src_tokenizer_dir = src_tokenizer_dir
        self.tgt_tokenizer_dir = tgt_tokenizer_dir
        self.encoder_sharing_groups = encoder_sharing_groups
        self.decoder_sharing_groups = decoder_sharing_groups
