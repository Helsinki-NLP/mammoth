from transformers import MarianConfig
from transformers import MarianTokenizer
import sentencepiece as spm

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts, build_bilingual_model

def _get_parser():
    parser = ArgumentParser(description='convert_mammoth_to_marian.py')
    train_opts(parser)
    build_bilingual_model(parser)
    parser.add_argument('--hf_output_dir')  # @TODO: uncomment this: , required=True)
    return parser


def initialize_marian_config(opt):
    return MarianConfig(
        vocab_size=opt.src_vocab_size,
        decoder_vocab_size=opt.src_vocab_size,
        max_position_embeddings=1024,  # default
        encoder_layers=sum(opt.enc_layers),
        encoder_ffn_dim=opt.transformer_ff,
        encoder_attention_heads=opt.heads,
        decoder_layers=sum(opt.dec_layers),
        decoder_ffn_dim=opt.transformer_ff,
        decoder_attention_heads=opt.heads,
        encoder_layerdrop=0.0,  # default
        decoder_layerdrop=0.0,  # default
        use_cache=True,  # default
        is_encoder_decoder=True,  # default
        activation_function=opt.pos_ffn_activation_fn,  # default
        d_model=opt.rnn_size,  # default
        dropout=opt.dropout,  # default
        attention_dropout=0,  # default
        activation_dropout=0,  # default
        init_std=0.02,  # default
        decoder_start_token_id=58100,  # default
        scale_embedding=False,  # default
        pad_token_id=3,  # hardcoded
        eos_token_id=1,  # hardcoded
        forced_eos_token_id=1,  # hardcoded
        share_encoder_decoder_embeddings=True,  # default
    )


def initialize_marian_tokenizer(opt) -> MarianTokenizer:
    def init_spm(vocabs_str: str, lang: str):
        def generate_subwords(vocab_path):
            with open(vocab_path, "r") as vocab_in:
                for word in vocab_in:
                    yield word
        sp = spm.SentencePieceProcessor()
        vocabs = eval(vocabs_str)
        # @TODO: feed `subwords` to spm
        subwords = list(generate_subwords(vocabs[lang]))
        return sp
    # @TODO: create an SP model -> instantiate the MarianTokenizer
    # initialize the SPM properly with src/tgt vocabularies provided by
    # the config file and src/tgt languages
    src_spm = init_spm(opt.src_vocab, opt.src_lang)
    try:
        tgt_spm = init_spm(opt.tgt_vocab, opt.tgt_lang)
    except:
        tgt_spm = src_spm
    return MarianTokenizer(
        src_spm,
        tgt_spm,
        source_lang=opt.src_lang,
        target_lang=opt.tgt_lang,
        unk_token=2,
        eos_token=1,
        pad_token=3,
        model_max_length=opt.src_seq_length,
    )


def convert_mammoth_to_marian() -> None:
    """
    """
    parser = _get_parser()
    opt, unknown = parser.parse_known_args()
    config = initialize_marian_config(opt)
    tokenizer = initialize_marian_tokenizer(opt)


if __name__ == "__main__":
    convert_mammoth_to_marian()
