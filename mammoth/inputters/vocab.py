import codecs
import collections
import itertools
import os

from mammoth.utils.logging import logger
from mammoth.constants import DefaultTokens

try:
    from tokenizers import Tokenizer
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False


DEFAULT_SPECIALS = (
    DefaultTokens.BOS,
    DefaultTokens.PAD,
    DefaultTokens.EOS,
    DefaultTokens.UNK,
)


def get_vocab(path, lang, size, specials=DEFAULT_SPECIALS, use_hf_tokenizer=False):
    """
    Factory function to load either traditional MAMMOTH vocab or HuggingFace tokenizer.

    Args:
        path: Path to vocab file (.txt) or tokenizer file (.json)
        lang: Language tag for logging
        size: Vocabulary size (ignored for HF tokenizers)
        specials: Special tokens (ignored for HF tokenizers)
        use_hf_tokenizer: If True, load as HuggingFace tokenizer

    Returns:
        Vocab or HFTokenizerVocab instance
    """
    if use_hf_tokenizer or path.endswith('.json'):
        new_vocab = HFTokenizerVocab(tokenizer_path=path, tag=lang)
    else:
        new_vocab = Vocab(path, items=None, tag=lang, size=size, specials=list(specials))

    logger.debug(new_vocab)
    return new_vocab


class Vocab:
    def __init__(self, path, items=None, tag="", size=None, specials=[]):
        if items is None:
            items, has_count = _read_vocab_file(path, tag)
            if has_count:
                items = sorted(items, key=lambda it: it[-1], reverse=True)
                items, _ = zip(*items)
                items = list(items)

        self.path = path
        size = None if size is None else size + len(specials)
        self.stoi = collections.defaultdict(itertools.count().__next__)
        self.itos = {self.stoi[elem]: elem for elem in (specials + items)[:size]}
        self.stoi = dict(self.stoi)
        self.specials = {elem: self.stoi[elem] for elem in specials}

    def __getitem__(self, key_str):
        return self.stoi[key_str]

    def __len__(self):
        return len(self.stoi)

    # TODO: likely can be deleted
    def add_token(self, token_str, is_special=False):
        if token_str not in self.stoi:
            idx = max(self.itos) + 1
            self.itos[idx] = token_str
            self.stoi[token_str] = idx
            if is_special:
                self.specials[token_str] = self.stoi[token_str]

    @classmethod
    def merge(cls, *vocabs, size=None):
        """Merge vocabs."""
        specials = collections.OrderedDict()
        for vocab in vocabs:
            for elem in vocab.specials:
                specials[elem] = None
        specials = [elem for elem in specials.keys()]

        # from itertools recipes https://docs.python.org/3/library/itertools.html#itertools-recipes
        # FIXME: would rather install more-itertools, but for now I'm trying to keep deps as minimal as possible
        def roundrobin(*iterables):
            "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
            # Recipe credited to George Sakkis
            num_active = len(iterables)
            nexts = itertools.cycle(iter(it).__next__ for it in iterables)
            while num_active:
                try:
                    for next in nexts:
                        yield next()
                except StopIteration:
                    # Remove the iterator we just exhausted from the cycle.
                    num_active -= 1
                    nexts = itertools.cycle(itertools.islice(nexts, num_active))

        items = list(roundrobin(*[vocab.stoi.keys() for vocab in vocabs]))
        return cls(None, items=items, tag="", size=size, specials=specials)

    def __repr__(self):
        return f"{self.__class__.__name__} @ {self.path} ({len(self)} items, specials=[{sorted(self.specials.keys())}])"


class HFTokenizerVocab:
    """Wrapper for HuggingFace tokenizers that provides MAMMOTH Vocab interface."""

    def __init__(self, tokenizer_path, tag=""):
        if not HF_TOKENIZERS_AVAILABLE:
            raise RuntimeError(
                "HuggingFace tokenizers library not available. "
                "Install with: pip install tokenizers"
            )

        logger.info(f"Loading {tag} HuggingFace tokenizer from {tokenizer_path}")

        if not os.path.exists(tokenizer_path):
            raise RuntimeError(f"{tag} tokenizer not found at {tokenizer_path}")

        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.path = tokenizer_path
        self.tag = tag

        # Build stoi (string to index) and itos (index to string) mappings
        vocab_dict = self.tokenizer.get_vocab()
        self.stoi = vocab_dict
        self.itos = {idx: token for token, idx in vocab_dict.items()}

        # Map MAMMOTH special tokens to HuggingFace tokens
        self.specials = {DefaultTokens.EOS: self.tokenizer.token_to_id("</s>"),
                         DefaultTokens.UNK: self.tokenizer.token_to_id("<unk>"),
                         DefaultTokens.BOS: self.tokenizer.token_to_id("<s>"),
                         DefaultTokens.PAD: self.tokenizer.token_to_id("<pad>"),
                         DefaultTokens.MASK: self.tokenizer.token_to_id("<mask>")}

        # Detect subword marker type by inspecting vocabulary
        # Most HF tokenizers use spacer (▁) for SentencePiece-style tokenization
        self.subword_type = self._detect_subword_type()

    def __getitem__(self, key_str):
        """Get token ID by token string (mimics Vocab behavior)."""
        token_id = self.tokenizer.token_to_id(key_str)
        if token_id is None:
            # Return UNK token ID if token not found
            unk_id = self.specials.get(DefaultTokens.UNK)
            if unk_id is None:
                raise KeyError(f"Token '{key_str}' not found and no UNK token defined")
            return unk_id
        return token_id

    def __len__(self):
        """Return vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def decode_token(self, token_id):
        """
        Decode a single token ID to its string representation.

        For BPE tokenizers, this returns the raw token string (e.g., 'Ġhello').
        For proper text decoding, use tokenizer.decode([ids]) instead.

        Args:
            token_id: Integer token ID

        Returns:
            Token string or '<unk>' if ID not in vocabulary
        """
        return self.itos.get(token_id, '<unk>')

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        """
        Decode a sequence of token IDs to text using the tokenizer's decoder.

        This properly handles BPE merging and special token removal.

        Args:
            token_ids: List of integer token IDs
            skip_special_tokens: Whether to remove special tokens from output

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _detect_subword_type(self):
        """
        Detect the subword marker type by inspecting the vocabulary.

        Returns:
            'sentencepiece' if spacer (▁) markers found
            'bpe' if joiner (￭) markers or Ġ prefix found
            'none' otherwise
        """
        from mammoth.constants import SubwordMarker

        # Sample tokens from vocabulary to check for markers
        sample_tokens = list(self.stoi.keys())[:100]

        has_spacer = any(SubwordMarker.SPACER in token for token in sample_tokens)
        has_joiner = any(SubwordMarker.JOINER in token for token in sample_tokens)
        has_gpt_marker = any(token.startswith('Ġ') for token in sample_tokens)

        if has_spacer:
            return 'sentencepiece'
        elif has_joiner or has_gpt_marker:
            return 'bpe'
        else:
            return 'none'

    def tokenize(self, text, is_train=False):
        """
        Tokenize text using the HuggingFace tokenizer.

        Args:
            text: Input text string or list of words
            is_train: Training mode flag (for future regularization support)

        Returns:
            List of token strings
        """
        if isinstance(text, list):
            # Join word list into text
            text = " ".join(text)

        encoding = self.tokenizer.encode(text)
        return encoding.tokens

    def tokenize_example(self, example, is_train=False):
        """
        Tokenize both src and tgt in an example dict.

        Args:
            example: Dict with 'src' and 'tgt' keys containing text
            is_train: Training mode flag

        Returns:
            Modified example dict with tokenized src and tgt
        """
        if 'src' in example and example['src'] is not None:
            example['src'] = self.tokenize(example['src'], is_train)

        if 'tgt' in example and example['tgt'] is not None:
            example['tgt'] = self.tokenize(example['tgt'], is_train)

        return example

    def __repr__(self):
        return (
            f"{self.__class__.__name__} @ {self.path} "
            f"({len(self)} items, subword_type={self.subword_type}, "
            f"specials={sorted(self.specials.keys())})"
        )


def create_vocabs_dict_from_hf_tokenizer(src_tokenizer_path, tgt_tokenizer_paths=None, src_lang="en", tgt_langs=None):
    """
    Create vocabs_dict from HuggingFace tokenizer(s) - supports both shared and separate vocabs

    Args:
        src_tokenizer_path: Path to source tokenizer.json file
        tgt_tokenizer_paths: Dict mapping language codes to tokenizer paths (e.g., {"ar": "path/to/ar_tokenizer.json"})
                            If None, uses src_tokenizer_path for all languages (multilingual setup)
        src_lang: Source language (default: "en")
        tgt_langs: List of target languages (default: ["ar"])

    Returns:
        Dict mapping (side, lang) tuples to HFTokenizerVocab objects
    """
    if tgt_langs is None:
        tgt_langs = ["ar"]

    if not os.path.exists(src_tokenizer_path):
        raise FileNotFoundError(
            f"Source tokenizer not found at {src_tokenizer_path}"
        )

    if not HF_TOKENIZERS_AVAILABLE:
        raise ImportError(
            "HuggingFace tokenizers library is not available. "
            "Please install it with: pip install tokenizers"
        )

    # Create source vocabulary
    src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
    src_vocab = HFTokenizerVocab(
        tokenizer_path=src_tokenizer_path,
        tag=f"src_{src_lang}",
    )
    vocabs_dict = {("src", src_lang): src_vocab}

    logger.info(f"Created source vocab: {len(src_vocab)} tokens from {src_tokenizer_path}")

    # Create target vocabularies
    if tgt_tokenizer_paths is None:
        # Multilingual setup: all languages share the same tokenizer
        logger.info(f"Using shared tokenizer for all languages (multilingual mode)")
        for tgt_lang in tgt_langs:
            tgt_vocab = HFTokenizerVocab(
                tokenizer_path=src_tokenizer_path,
                tag=f"tgt_{tgt_lang}",
            )
            vocabs_dict[("tgt", tgt_lang)] = tgt_vocab
    else:
        # Separate vocabs: each target language has its own tokenizer
        logger.info(f"Using separate tokenizers for target languages:")
        for tgt_lang in tgt_langs:
            if tgt_lang not in tgt_tokenizer_paths:
                raise ValueError(
                    f"Target language '{tgt_lang}' not found in tgt_tokenizer_paths. "
                    f"Available: {list(tgt_tokenizer_paths.keys())}"
                )

            tgt_path = tgt_tokenizer_paths[tgt_lang]
            if not os.path.exists(tgt_path):
                raise FileNotFoundError(f"Target tokenizer not found at {tgt_path}")

            tgt_tokenizer = Tokenizer.from_file(tgt_path)
            tgt_vocab = HFTokenizerVocab(
                tokenizer_path=tgt_path,
                tag=f"tgt_{tgt_lang}",
            )
            vocabs_dict[("tgt", tgt_lang)] = tgt_vocab
            logger.info(f"  - {tgt_lang}: {len(tgt_vocab)} tokens ({tgt_path})")

    return vocabs_dict


def _read_vocab_file(vocab_path, tag):
    """Loads a vocabulary from the given path.
    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line, may followed with a count number
            seperate by space if `with_count`. No extra whitespace is allowed.
        tag (str): Used for logging which vocab is being read.
    """

    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError("{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, "r", "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            first_line = lines[0].split(None, 1)
            has_count = len(first_line) == 2 and first_line[-1].isdigit()
            if has_count:
                vocab = [line.split(None, 1) for line in lines]
                orig_len = len(vocab)
                vocab = [tpl for tpl in vocab if len(tpl) == 2]
                if len(vocab) != orig_len:
                    logger.warning(f"Dropped invalid entries from {vocab_path}")
            else:
                vocab = [line.strip().split()[0] for line in lines]
            return vocab, has_count
