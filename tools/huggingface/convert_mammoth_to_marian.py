from transformers import MarianConfig
import sys
import yaml

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts


def _get_parser():
    parser = ArgumentParser(description='convert_mammoth_to_marian.py')
    train_opts(parser)
    parser.add_argument('hf_ckpt_dir', required=True)
    return parser





def convert_mammoth_to_marian(mammoth_config_path: str, marian_config_path: str) -> None:
    """
    """
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()

    config_dict = opt # yaml.safe_load(mammoth_config)
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
            activation_function=config_dict['pos_ffn_activation_fn'],  # default
            d_model=config_dict['rnn_size'],  # default
            dropout=confiog_dict['dropout'],  # default
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_mammoth_to_marian(**args.__dict__)
