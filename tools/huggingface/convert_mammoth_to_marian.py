from transformers import MarianConfig
from transformers import MarianTokenizer
import sentencepiece as spm

from onmt.utils.parse import ArgumentParser
from onmt.constants import DefaultTokens
import os
import json
from onmt.opts import train_opts, build_bilingual_model

def _get_parser():
    parser = ArgumentParser(description='convert_mammoth_to_marian.py')
    train_opts(parser)
    build_bilingual_model(parser)
    parser.add_argument('--hf_output_dir')  # @TODO: uncomment this: , required=True)
    parser.add_argument("--src_spm_model")
    parser.add_argument("--src_spm_vocab")
    parser.add_argument("--tgt_spm_model")
    parser.add_argument("--tgt_spm_vocab")
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
    def init_spm(vocabs_str: str, lang: str) -> None:
        def generate_subwords(vocab_path):
            with open(vocab_path, "r") as vocab_in:
                for word in vocab_in:
                    yield word
        sp = spm.SentencePieceProcessor()
        vocabs = eval(vocabs_str)
        # @TODO: feed `subwords` to spm
        subwords = list(generate_subwords(vocabs[lang]))
        return sp

    def read_vocab(spm_vocab_path: str) -> str:
        json_vocab = {}
        json_vocab_path = os.path.join(os.path.dirname(spm_vocab_path), "vocab.json")
        with open(spm_vocab_path, "r") as spm_vocab_in:
            for idx, subword in enumerate(spm_vocab_in):
                if "\t" in subword:
                    subword = subword.split("\t")[0]
                json_vocab[subword] = idx
        with open(json_vocab_path, "w") as json_out:
            json.dump(json_vocab, json_out)
        return json_vocab_path

    src_spm, tgt_spm, vocab = "", "", ""
    if not opt.src_spm_model:
        # create an SP model -> instantiate the MarianTokenizer
        # @TODO: initialize the SPM properly with src/tgt vocabularies provided by
        # the config file and src/tgt languages
        src_spm = init_spm(opt.src_vocab, opt.src_lang)
        # @TODO: get vocab?
    else:
        src_spm = opt.src_spm_model
    if not opt.target_spm_model:
        try:
            tgt_spm = init_spm(opt.tgt_vocab, opt.tgt_lang)
        except:
            tgt_spm = src_spm
        # @TODO: MarianTokenizer requires the spm paths, so we need to save these SPMs
    else:
        tgt_spm = opt.tgt_spm_model

    if not vocab:
        vocab = read_vocab(opt.source_spm_vocab)

    return MarianTokenizer(
        source_spm=src_spm,
        target_spm=tgt_spm,
        vocab=vocab,
        source_lang=opt.src_lang,
        target_lang=opt.tgt_lang,
        unk_token=DefaultTokens.UNK,
        eos_token=DefaultTokens.EOS,
        pad_token=DefaultTokens.PAD,
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
