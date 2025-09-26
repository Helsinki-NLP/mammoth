"""Transforms relate to tokenization/subword."""

from mammoth.utils.logging import logger
from mammoth.transforms import register_transform
from mammoth.constants import DefaultTokens, SubwordMarker
from .transform import Transform, ObservableStats


class TokenizerTransform(Transform):
    """Tokenizer transform abstract class."""

    def __init__(self, opts):
        """Initialize necessary options for Tokenizer."""
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relate to Subword."""
        # Sharing options among `TokenizerTransform`s, same name conflict in
        # this scope will be resolved by remove previous occurrence in parser
        group = parser.add_argument_group(
            "Transform/Subword/Common",
            conflict_handler="resolve",
            description=".. Attention:: Common options shared by all subword transforms. "  # noqa: E501
            "Including options for indicate subword model path, "
            "`Subword Regularization <https://arxiv.org/abs/1804.10959>`_"
            "/`BPE-Dropout <https://arxiv.org/abs/1910.13267>`_, "
            "and `Vocabulary Restriction "
            "<https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt>`__.",
        )  # noqa: E501
        group.add(
            "-src_subword_model",
            "--src_subword_model",
            help="Path of subword model for src (or shared).",
        )
        group.add(
            "-tgt_subword_model",
            "--tgt_subword_model",
            help="Path of subword model for tgt.",
        )

        # subword regularization(or BPE dropout) options:
        group.add(
            "-src_subword_nbest",
            "--src_subword_nbest",
            type=int,
            default=1,
            help="Number of candidates in subword regularization. "
            "Valid for unigram sampling, "
            "invalid for BPE-dropout. "
            "(source side)",
        )
        group.add(
            "-tgt_subword_nbest",
            "--tgt_subword_nbest",
            type=int,
            default=1,
            help="Number of candidates in subword regularization. "
            "Valid for unigram sampling, "
            "invalid for BPE-dropout. "
            "(target side)",
        )
        group.add(
            "-src_subword_alpha",
            "--src_subword_alpha",
            type=float,
            default=0,
            help="Smoothing parameter for sentencepiece unigram "
            "sampling, and dropout probability for BPE-dropout. "
            "(source side)",
        )
        group.add(
            "-tgt_subword_alpha",
            "--tgt_subword_alpha",
            type=float,
            default=0,
            help="Smoothing parameter for sentencepiece unigram "
            "sampling, and dropout probability for BPE-dropout. "
            "(target side)",
        )
        group.add(
            "-src_subword_type",
            "--src_subword_type",
            type=str,
            default="sentencepiece",
            choices=["sentencepiece", "bpe", "none"],
            help="Type of subword tokenization for source side. "
            "Used by denoising transforms to determine word boundaries.",
        )
        group.add(
            "-tgt_subword_type",
            "--tgt_subword_type",
            type=str,
            default="sentencepiece",
            choices=["sentencepiece", "bpe", "none"],
            help="Type of subword tokenization for target side. "
            "Used by denoising transforms to determine word boundaries.",
        )

        # subword vocabulary restriction options:
        group.add(
            "-src_subword_vocab",
            "--src_subword_vocab",
            type=str,
            default="",
            help="Path to the vocabulary file for src subword. Format: <word>\t<count> per line.",
        )
        group.add(
            "-tgt_subword_vocab",
            "--tgt_subword_vocab",
            type=str,
            default="",
            help="Path to the vocabulary file for tgt subword. Format: <word>\t<count> per line.",
        )
        group.add(
            "-src_vocab_threshold",
            "--src_vocab_threshold",
            type=int,
            default=0,
            help="Only produce src subword in src_subword_vocab with frequency >= src_vocab_threshold.",
        )
        group.add(
            "-tgt_vocab_threshold",
            "--tgt_vocab_threshold",
            type=int,
            default=0,
            help="Only produce tgt subword in tgt_subword_vocab with frequency >= tgt_vocab_threshold.",
        )
        group.add(
            "-share_vocab",
            "--share_vocab",
            action="store_true",
            help="use the same model for both sides",
        )

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks for Subword options."""
        assert (
            0 <= opts.src_subword_alpha <= 1
        ), "src_subword_alpha should be in the range [0, 1]"
        assert (
            0 <= opts.tgt_subword_alpha <= 1
        ), "tgt_subword_alpha should be in the range [0, 1]"

    def _parse_opts(self):
        self.share_vocab = self.opts.share_vocab
        self.src_subword_model = self.opts.src_subword_model
        self.tgt_subword_model = self.opts.tgt_subword_model
        self.src_subword_nbest = self.opts.src_subword_nbest
        self.tgt_subword_nbest = self.opts.tgt_subword_nbest
        self.src_subword_alpha = self.opts.src_subword_alpha
        self.tgt_subword_alpha = self.opts.tgt_subword_alpha
        self.src_subword_vocab = self.opts.src_subword_vocab
        self.tgt_subword_vocab = self.opts.tgt_subword_vocab
        self.src_vocab_threshold = self.opts.src_vocab_threshold
        self.tgt_vocab_threshold = self.opts.tgt_vocab_threshold

    def _repr_args(self):
        """Return str represent key arguments for TokenizerTransform."""
        kwargs = {
            "share_vocab": self.share_vocab,
            "src_subword_model": self.src_subword_model,
            "tgt_subword_model": self.tgt_subword_model,
            "src_subword_alpha": self.src_subword_alpha,
            "tgt_subword_alpha": self.tgt_subword_alpha,
            "src_subword_vocab": self.src_subword_vocab,
            "tgt_subword_vocab": self.tgt_subword_vocab,
            "src_vocab_threshold": self.src_vocab_threshold,
            "tgt_vocab_threshold": self.tgt_vocab_threshold,
        }
        return ", ".join([f"{kw}={arg}" for kw, arg in kwargs.items()])


class SubwordStats(ObservableStats):
    """Runing statistics for counting tokens before/after subword transform."""

    __slots__ = ["subwords", "words"]

    def __init__(self, subwords: int, words: int):
        self.subwords = subwords
        self.words = words

    def update(self, other: "SubwordStats"):
        self.subwords += other.subwords
        self.words += other.words

    def __str__(self) -> str:
        return "{}: {} -> {} tokens".format(self.name(), self.words, self.subwords)


@register_transform(name="sentencepiece")
class SentencePieceTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import sentencepiece as spm

        spm.set_random_generator_seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        import sentencepiece as spm

        load_src_model = spm.SentencePieceProcessor()
        if self.task:
            concrete_model = self.src_subword_model.format(
                src_lang=self.task.src_lang,
                tgt_lang=self.task.tgt_lang,
            )
        else:
            concrete_model = self.src_subword_model
        logger.info(f"Concrete SentencePiece model, src: {concrete_model}")
        load_src_model.Load(concrete_model)
        _diff_vocab = (
            self.src_subword_vocab != self.tgt_subword_vocab
            or self.src_vocab_threshold != self.tgt_vocab_threshold
        )
        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            load_src_model.LoadVocabulary(
                self.src_subword_vocab, self.src_vocab_threshold
            )
        if self.share_vocab and not _diff_vocab:
            self.load_models = {"src": load_src_model, "tgt": load_src_model}
        else:
            load_tgt_model = spm.SentencePieceProcessor()
            if self.task:
                concrete_model = self.tgt_subword_model.format(
                    src_lang=self.task.src_lang,
                    tgt_lang=self.task.tgt_lang,
                )
            else:
                concrete_model = self.tgt_subword_model
            logger.info(f"Concrete SentencePiece model, tgt: {concrete_model}")
            load_tgt_model.Load(concrete_model)
            if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
                load_tgt_model.LoadVocabulary(
                    self.tgt_subword_vocab, self.tgt_vocab_threshold
                )
            self.load_models = {"src": load_src_model, "tgt": load_tgt_model}

    def _tokenize(self, tokens, side="src", is_train=False):
        """Do sentencepiece subword tokenize."""
        sp_model = self.load_models[side]
        sentence = " ".join(tokens)
        nbest_size = self.tgt_subword_nbest if side == "tgt" else self.src_subword_nbest
        if is_train is False or nbest_size in [0, 1]:
            # derterministic subwording
            segmented = sp_model.encode(sentence, out_type=str)
        else:
            # subword sampling when nbest_size > 1 or -1
            # alpha should be 0.0 < alpha < 1.0
            alpha = self.tgt_subword_alpha if side == "tgt" else self.src_subword_alpha
            segmented = sp_model.encode(
                sentence,
                out_type=str,
                enable_sampling=True,
                alpha=alpha,
                nbest_size=nbest_size,
            )
        return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize(example["src"], "src", is_train)
        tgt_out = (
            self._tokenize(example["tgt"], "tgt", is_train)
            if example["tgt"] is not None
            else None
        )
        if stats is not None:
            tgt_len_in = 0 if example["tgt"] is None else len(example["tgt"])
            tgt_len_out = 0 if tgt_out is None else len(tgt_out)
            n_words = len(example["src"]) + tgt_len_in
            n_subwords = len(src_out) + tgt_len_out
            stats.update(SubwordStats(n_subwords, n_words))
        example["src"], example["tgt"] = src_out, tgt_out
        return example

    def apply_reverse(self, translated):
        """Reverse the SentencePiece tokenization by detokenizing subwords."""
        if isinstance(translated, str):
            tokens = translated.split()
        elif isinstance(translated, list):
            tokens = translated
        else:
            return translated

        # Define special tokens to remove
        special_tokens = [
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
            DefaultTokens.UNK,
            DefaultTokens.MASK,
        ]

        # Remove special tokens
        cleaned_tokens = [token for token in tokens if token not in special_tokens]

        # Handle SentencePiece detokenization
        detokenized_text = ""
        for token in cleaned_tokens:
            if token.startswith(SubwordMarker.SPACER):  # ▁ indicates word boundary
                # Remove ▁ and add space (except for first token)
                if detokenized_text:
                    detokenized_text += " " + token[1:]
                else:
                    detokenized_text += token[
                        1:
                    ]  # First token doesn't need leading space
            else:
                # No ▁ means it's a continuation of the previous word
                detokenized_text += token
        detokenized_text = detokenized_text.strip()
        logger.info(f"Translated tokens: '{translated}'")
        logger.info(f"Final detokenized text: '{detokenized_text}'")

        return detokenized_text

    def _repr_args(self):
        """Return str represent key arguments for class."""
        kwargs_str = super()._repr_args()
        additional_str = "src_subword_nbest={}, tgt_subword_nbest={}".format(
            self.src_subword_nbest, self.tgt_subword_nbest
        )
        return kwargs_str + ", " + additional_str


@register_transform(name="bpe")
class BPETransform(TokenizerTransform):
    """subword_nmt: official BPE subword transform class."""

    def __init__(self, opts):
        """Initialize necessary options for subword_nmt."""
        super().__init__(opts)

    def _parse_opts(self):
        super()._parse_opts()
        self.dropout = {"src": self.src_subword_alpha, "tgt": self.tgt_subword_alpha}

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import random

        random.seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        from subword_nmt.apply_bpe import BPE, read_vocabulary

        # Load vocabulary file if provided and set threshold
        src_vocabulary, tgt_vocabulary = None, None
        if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
            with open(self.src_subword_vocab, encoding="utf-8") as _sv:
                src_vocabulary = read_vocabulary(_sv, self.src_vocab_threshold)
        if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
            with open(self.tgt_subword_vocab, encoding="utf-8") as _tv:
                tgt_vocabulary = read_vocabulary(_tv, self.tgt_vocab_threshold)
        # Load Subword Model
        with open(self.src_subword_model, encoding="utf-8") as src_codes:
            load_src_model = BPE(codes=src_codes, vocab=src_vocabulary)
        if self.share_vocab and (src_vocabulary == tgt_vocabulary):
            self.load_models = {"src": load_src_model, "tgt": load_src_model}
        else:
            with open(self.tgt_subword_model, encoding="utf-8") as tgt_codes:
                load_tgt_model = BPE(codes=tgt_codes, vocab=tgt_vocabulary)
            self.load_models = {"src": load_src_model, "tgt": load_tgt_model}

    def _tokenize(self, tokens, side="src", is_train=False):
        """Do bpe subword tokenize."""
        bpe_model = self.load_models[side]
        dropout = self.dropout[side] if is_train else 0.0
        segmented = bpe_model.segment_tokens(tokens, dropout=dropout)
        return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply bpe subword encode to src & tgt."""
        src_out = self._tokenize(example["src"], "src", is_train)
        tgt_out = self._tokenize(example["tgt"], "tgt", is_train)
        if stats is not None:
            n_words = len(example["src"]) + len(example["tgt"])
            n_subwords = len(src_out) + len(tgt_out)
            stats.update(SubwordStats(n_subwords, n_words))
        example["src"], example["tgt"] = src_out, tgt_out
        return example


# Discontinue supporting OpenNMT Tokenizer
# @register_transform(name="onmt_tokenize")


@register_transform(name="huggingface")
class HuggingFaceTokenizerTransform(TokenizerTransform):
    """HuggingFace tokenizer transform class."""

    def __init__(self, opts):
        """Initialize necessary options for HuggingFace tokenizer."""
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options for HuggingFace tokenizer."""
        super().add_options(parser)
        group = parser.add_argument_group("Transform/Subword/HuggingFace")

        # Model specification options
        group.add(
            "-src_hf_model_name",
            "--src_hf_model_name",
            type=str,
            default="",
            help="HuggingFace model name/path for src tokenizer (e.g., 'bert-base-uncased', 'facebook/bart-large')",
        )
        group.add(
            "-tgt_hf_model_name",
            "--tgt_hf_model_name",
            type=str,
            default="",
            help="HuggingFace model name/path for tgt tokenizer",
        )

        # Tokenizer configuration options
        group.add(
            "-src_hf_tokenizer_kwargs",
            "--src_hf_tokenizer_kwargs",
            type=str,
            default="{}",
            help="Additional kwargs for src HuggingFace tokenizer in dict string format (e.g., \"{'add_special_tokens': False}\")",
        )
        group.add(
            "-tgt_hf_tokenizer_kwargs",
            "--tgt_hf_tokenizer_kwargs",
            type=str,
            default="{}",
            help="Additional kwargs for tgt HuggingFace tokenizer in dict string format",
        )

        # Encoding/decoding options
        group.add(
            "-hf_add_special_tokens",
            "--hf_add_special_tokens",
            action="store_true",
            help="Whether to add special tokens during tokenization (BOS, EOS, etc.)",
        )
        group.add(
            "-hf_return_token_type_ids",
            "--hf_return_token_type_ids",
            action="store_true",
            help="Whether to return token type IDs",
        )
        group.add(
            "-hf_return_attention_mask",
            "--hf_return_attention_mask",
            action="store_true",
            help="Whether to return attention mask",
        )
        group.add(
            "-hf_max_length",
            "--hf_max_length",
            type=int,
            default=None,
            help="Maximum length for tokenization (None for no limit)",
        )
        group.add(
            "-hf_padding",
            "--hf_padding",
            type=str,
            default="do_not_pad",
            choices=["do_not_pad", "longest", "max_length"],
            help="Padding strategy for tokenization",
        )
        group.add(
            "-hf_truncation",
            "--hf_truncation",
            action="store_true",
            help="Whether to truncate sequences that exceed max_length",
        )

    @classmethod
    def _validate_options(cls, opts):
        """Extra checks for HuggingFace tokenizer options."""
        super()._validate_options(opts)

        # Check that at least src model is specified
        if not opts.src_hf_model_name:
            raise ValueError("src_hf_model_name is required for HuggingFace tokenizer")

        # Validate kwargs dictionaries
        try:
            src_kwargs_dict = eval(opts.src_hf_tokenizer_kwargs)
            if not isinstance(src_kwargs_dict, dict):
                raise ValueError("-src_hf_tokenizer_kwargs must be a valid dict string")
        except Exception as e:
            raise ValueError(f"Invalid src_hf_tokenizer_kwargs: {e}")

        try:
            tgt_kwargs_dict = eval(opts.tgt_hf_tokenizer_kwargs)
            if not isinstance(tgt_kwargs_dict, dict):
                raise ValueError("-tgt_hf_tokenizer_kwargs must be a valid dict string")
        except Exception as e:
            raise ValueError(f"Invalid tgt_hf_tokenizer_kwargs: {e}")

    def _parse_opts(self):
        """Parse options specific to HuggingFace tokenizer."""
        super()._parse_opts()

        # Model names
        self.src_hf_model_name = self.opts.src_hf_model_name
        self.tgt_hf_model_name = self.opts.tgt_hf_model_name or self.src_hf_model_name

        # Additional tokenizer kwargs
        self.src_hf_tokenizer_kwargs = eval(self.opts.src_hf_tokenizer_kwargs)
        self.tgt_hf_tokenizer_kwargs = eval(self.opts.tgt_hf_tokenizer_kwargs)

        # Encoding options
        self.hf_add_special_tokens = self.opts.hf_add_special_tokens
        self.hf_return_token_type_ids = self.opts.hf_return_token_type_ids
        self.hf_return_attention_mask = self.opts.hf_return_attention_mask
        self.hf_max_length = self.opts.hf_max_length
        self.hf_padding = self.opts.hf_padding
        self.hf_truncation = self.opts.hf_truncation

    def warm_up(self, vocabs=None):
        """Load HuggingFace tokenizers."""
        super().warm_up(None)

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace transformers library is required for HuggingFace tokenizer. "
                "Install it with: pip install transformers"
            )

        logger.info(f"Loading HuggingFace tokenizer from {self.src_hf_model_name}")

        # Load source tokenizer
        src_tokenizer = AutoTokenizer.from_pretrained(
            self.src_hf_model_name, **self.src_hf_tokenizer_kwargs
        )

        # Load target tokenizer (same as source if not specified)
        if self.share_vocab or self.tgt_hf_model_name == self.src_hf_model_name:
            self.load_models = {"src": src_tokenizer, "tgt": src_tokenizer}
            logger.info("Using shared tokenizer for both src and tgt")
        else:
            # logger.info(f"Loading separate target tokenizer from {self.tgt_hf_model_name}")
            tgt_tokenizer = AutoTokenizer.from_pretrained(
                self.tgt_hf_model_name, **self.tgt_hf_tokenizer_kwargs
            )
            self.load_models = {"src": src_tokenizer, "tgt": tgt_tokenizer}

    def _tokenize(self, tokens, side="src", is_train=False):
        """Tokenize using HuggingFace tokenizer."""
        tokenizer = self.load_models[side]
        text = " ".join(tokens) if isinstance(tokens, list) else tokens
        return tokenizer.tokenize(text)

    def _detokenize(self, tokens, side="src"):
        """Detokenize using HuggingFace tokenizer."""
        tokenizer = self.load_models[side]

        # Convert tokens to IDs if they're strings
        if isinstance(tokens[0], str):
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
        else:
            token_ids = tokens
        # Decode back to text
        text = tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Return as list of tokens for consistency
        return text.split()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply HuggingFace tokenization to src & tgt."""
        src_out = self._tokenize(example["src"], "src", is_train)
        tgt_out = (
            self._tokenize(example["tgt"], "tgt", is_train)
            if example["tgt"] is not None
            else None
        )

        if stats is not None:
            tgt_len_in = 0 if example["tgt"] is None else len(example["tgt"])
            tgt_len_out = 0 if tgt_out is None else len(tgt_out)
            n_words = len(example["src"]) + tgt_len_in
            n_subwords = len(src_out) + tgt_len_out
            stats.update(SubwordStats(n_subwords, n_words))

        example["src"], example["tgt"] = src_out, tgt_out
        return example

    def apply_reverse(self, translated):
        """Reverse transform: detokenize the translated output."""
        # logger.info(f"Input to apply_reverse: {translated} (type: {type(translated)})")

        if isinstance(translated, str):
            # Split the string into tokens first
            logger.info(f"translated tokens: '{translated}'")
            tokens = translated.split()
            # logger.info(f"Split into tokens: {tokens}")
        elif isinstance(translated, list):
            tokens = translated
        else:
            logger.warning(f"Unexpected input type {type(translated)}, returning as-is")
            return translated

        # Define special tokens to remove
        special_tokens = [
            DefaultTokens.PAD,
            DefaultTokens.BOS,
            DefaultTokens.EOS,
            DefaultTokens.UNK,
            DefaultTokens.MASK,
        ]
        # special_tokens = {'<pad>', '<s>', '</s>', '<unk>', '<mask>'}

        # Remove special tokens
        cleaned_tokens = [token for token in tokens if token not in special_tokens]
        # logger.info(f"After removing special tokens: {cleaned_tokens}")

        # Handle SentencePiece tokens (▁ prefix indicates word boundaries)
        detokenized_text = ""
        for token in cleaned_tokens:
            if token.startswith("▁") or token.startswith("Ġ"):
                # ▁ and Ġ indicate start of a new word, replace with space
                detokenized_text += " " + token[1:]  # Remove ▁ and Ġ and add space
            else:
                # No ▁ or Ġ means it's a continuation of the previous word
                detokenized_text += token

        # Clean up extra spaces and strip
        detokenized_text = detokenized_text.strip()
        logger.info(f"Final detokenized text: '{detokenized_text}'")

        # Return as list of words for consistency with mammoth's expectations
        return detokenized_text

    def _repr_args(self):
        """Return string representation of key arguments."""
        kwargs_str = super()._repr_args()
        additional_str = f"src_hf_model_name={self.src_hf_model_name}, tgt_hf_model_name={self.tgt_hf_model_name}"
        additional_str += f", hf_add_special_tokens={self.hf_add_special_tokens}"
        return kwargs_str + ", " + additional_str
