from transformers import MarianConfig
import sys
import yaml
import argparse


def convert_mammoth_to_marian(mammoth_config_path: str, marian_config_path: str) -> None:
    """
    """
    with (
        open(mammoth_config_path, "r") if mammoth_config_path is not sys.stdin else sys.stdin as mammoth_config,
        open(marian_config_path, "w") if marian_config_path is not sys.stdout else sys.stdout as marian_config
    ):
        config_dict = yaml.safe_load(mammoth_config)
        marian = MarianConfig(
            vocab_size=config_dict["src_vocab_size"],
            decoder_vocab_size=config_dict["src_vocab_size"],
            max_position_embeddings=1024,  # default
            encoder_layers=config_dict["enc_layers"][0],
            encoder_ffn_dim=config_dict["transformer_ff"],
            encoder_attention_heads=config_dict["heads"],
            decoder_layers=config_dict["dec_layers"][0],
            decoder_ffn_dim=config_dict["transformer_ff"],
            decoder_attention_heads=config_dict["heads"],
            encoder_layerdrop=0.0,  # default
            decoder_layerdrop=0.0,  # default
            use_cache=True,  # default
            is_encoder_decoder=True,  # default
            activation_function="gelu",  # default
            d_model=1024,  # default
            dropout=0.1,  # default
            attention_dropout=0,  # default
            activation_dropout=0,  # default
            init_std=0.02,  # default
            decoder_start_token_id=58100,  # default
            scale_embedding=False,  # default
            pad_token_id=58100,  # default
            eos_token_id=0,  # default
            forced_eos_token_id=0,  # default
            share_encoder_decoder_embeddings=True,  # default
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", "-s", dest="mammoth_config_path", default=sys.stdin)
    parser.add_argument("--tgt", "-t", dest="marian_config_path", default=sys.stdout)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_mammoth_to_marian(**args.__dict__)