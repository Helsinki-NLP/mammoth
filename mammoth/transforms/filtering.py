from mammoth.transforms import register_transform
from .transform import Transform, ObservableStats
from mammoth.utils.logging import logger
import re
import math
import itertools
import string
import difflib


class FilterTooLongStats(ObservableStats):
    """Runing statistics for FilterTooLongTransform."""

    __slots__ = ["filtered"]

    def __init__(self):
        self.filtered = 1

    def update(self, other: "FilterTooLongStats"):
        self.filtered += other.filtered


@register_transform(name='filtertoolong')
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relating to this Transform."""
        group = parser.add_argument_group("Transform/Length filter")
        group.add("--src_seq_length", "-src_seq_length", type=int, default=200, help="Maximum source sequence length.")
        group.add("--tgt_seq_length", "-tgt_seq_length", type=int, default=200, help="Maximum target sequence length.")

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length
        self._hf_overhead_adjusted = False

    def warm_up(self, vocabs):
        """
        Adjust thresholds when using HFTokenizerVocab to account for special token overhead.

        When using HF tokenizers, numericalization adds [BOS, *tokens, EOS] = 2 extra tokens.
        We reduce thresholds by 2 to ensure final tensors fit within max_length.
        """
        from mammoth.inputters.vocab import HFTokenizerVocab

        logger.info(f"FilterTooLongTransform.warm_up() called with vocabs: {list(vocabs.keys())}")
        logger.info(f"  src vocab type: {type(vocabs.get('src', None))}")
        logger.info(f"  tgt vocab type: {type(vocabs.get('tgt', None))}")

        # Check if any vocab is an HF tokenizer
        uses_hf_tokenizer = any(isinstance(v, HFTokenizerVocab) for v in vocabs.values())
        logger.info(f"  uses_hf_tokenizer: {uses_hf_tokenizer}")

        if uses_hf_tokenizer and not self._hf_overhead_adjusted:
            # Overhead: 2 special tokens (BOS, EOS)
            # After fix to use direct token-to-ID lookup, no re-encoding variance
            overhead = 2

            original_src = self.src_seq_length
            original_tgt = self.tgt_seq_length

            self.src_seq_length = max(1, self.src_seq_length - overhead)
            self.tgt_seq_length = max(1, self.tgt_seq_length - overhead)

            self._hf_overhead_adjusted = True

            logger.info(
                f"✅ FilterTooLongTransform: Adjusted thresholds for HF tokenizer overhead. "
                f"src: {original_src} → {self.src_seq_length}, "
                f"tgt: {original_tgt} → {self.tgt_seq_length} "
                f"(reserves {overhead} tokens for special tokens + re-encoding)"
            )
        else:
            logger.info(f"FilterTooLongTransform: No adjustment needed. HF={uses_hf_tokenizer}, Already adjusted={self._hf_overhead_adjusted}")

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        src_len = len(example['src'])
        tgt_len = len(example['tgt'])
        if src_len == 0 or tgt_len == 0:
            # also filter empty strings
            return None
        if src_len > self.src_seq_length or tgt_len > self.tgt_seq_length:
            if stats is not None:
                stats.update(FilterTooLongStats())
            return None
        else:
            # Debug: Log sequences near the threshold that pass through
            threshold_margin = 5
            if src_len > self.src_seq_length - threshold_margin or tgt_len > self.tgt_seq_length - threshold_margin:
                logger.warning(
                    f"FilterTooLong: Sequence near threshold PASSED. "
                    f"src_len={src_len} (threshold={self.src_seq_length}), "
                    f"tgt_len={tgt_len} (threshold={self.tgt_seq_length}), "
                    f"ratio={tgt_len/src_len:.2f}. "
                    f"After numericalization will add ~3 special tokens."
                )
                logger.warning(f"  SRC tokens: {example['src'][:20]} ...")
                logger.warning(f"  TGT tokens: {example['tgt'][:20]} ...")
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format('src_seq_length', self.src_seq_length, 'tgt_seq_length', self.tgt_seq_length)


# Filters inspired by OpusFilter
# https://github.com/Helsinki-NLP/OpusFilter/blob/aca40bd064d9b087c5216de0568d7fb91a31d142/opusfilter/filters.py


@register_transform(name='filterwordratio')
class FilterWordRatio(Transform):
    """Filter out sentence based on word length ratio"""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relating to this Transform."""
        group = parser.add_argument_group("Transform/Word ratio filter")
        group.add("--word_ratio_threshold", "-word_ratio_threshold", type=int, default=3,
                  help="Threshold for discarding sentences based on word ratio.")

    def _parse_opts(self):
        self.word_ratio_threshold = self.opts.word_ratio_threshold

    def apply(self, example, **kwargs):
        """Return None if too long else return as is."""
        src_len = len(example['src'])
        tgt_len = len(example['tgt'])
        lengths = sorted([src_len, tgt_len])
        if lengths[0] == 0:
            return None
        else:
            ratio = lengths[-1] / lengths[0]
            if ratio < self.word_ratio_threshold:
                return example
            else:
                return None

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('word_ratio_threshold', self.word_ratio_threshold)


@register_transform(name='filterrepetitions')
class FilterRepetitions(Transform):
    """Filter segments with repeated content. Useful e.g. for filtering data generated by a low-quality NMT model."""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relating to this Transform."""
        group = parser.add_argument_group("Transform/Repetitions filter")
        group.add("--rep_threshold", "-rep_threshold", type=int, default=2,
                  help="Number of times the substring is repeated.")
        group.add("--rep_min_len", "-rep_min_len", type=int, default=3,
                  help="Minimum length of the repeated pattern.")
        group.add("--rep_max_len", "-rep_max_len", type=int, default=100,
                  help="Maximum length of the repeated pattern.")

    def _parse_opts(self):
        self.rep_threshold = self.opts.rep_threshold
        self.rep_min_len = self.opts.rep_min_len
        self.rep_max_len = self.opts.rep_max_len

    def apply(self, example, **kwargs):
        """Return None if the repeated pattern appears more than n-threshold times."""
        # compiled regexp for finding repetitions
        rstring = f'(\\S.{{{self.rep_min_len-1},{self.rep_max_len}}}?)(?: *\\1){{{self.rep_threshold},}}'
        regex = re.compile(rstring)
        reps = []
        for segment in example['src'], example['tgt']:
            match = regex.search(' '.join(segment))
            if match:
                full = match.group(0)
                repeated = match.group(1)
                rep = full.count(repeated) - 1
            else:
                rep = 0
            reps.append(rep)
        if max(reps) > self.rep_threshold:
            return None
        else:
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}, {}={}'.format('rep_threshold', self.rep_threshold,
                                            'rep_min_len', self.rep_min_len, 'rep_max_len', self.rep_max_len)


@register_transform(name='filterterminalpunct')
class FilterTerminalPunctuation(Transform):
    """Filter segments with respect to the co-occurrence of terminal punctuation marks"""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relating to this Transform."""
        group = parser.add_argument_group("Transform/Terminal punctuation filter")
        group.add("--punct_threshold", "-punct_threshold", type=int, default=-2,
                  help="Minimum penalty score for discarding sentences based on their terminal punctuation signs")

    def _parse_opts(self):
        self.punct_threshold = self.opts.punct_threshold

    def apply(self, example, **kwargs):
        """Return None if the penalty is smaller than the threshold."""
        src = ' '.join(example['src'])
        tgt = ' '.join(example['tgt'])
        spun = len([c for c in src if c in ['.', '?', '!', '…']])
        tpun = len([c for c in tgt if c in ['.', '?', '!', '…']])
        score = abs(spun - tpun)
        if spun > 1:
            score += spun - 1
        if tpun > 1:
            score += tpun - 1
        score = -math.log(score + 1)
        if score >= self.punct_threshold:
            return example
        else:
            return None

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('punct_threshold', self.punct_threshold)


@register_transform(name='filternonzeronumerals')
class FilterNonZeroNumerals(Transform):
    """Filter segments based on a similarity measure of numerals between the segments with zeros removed"""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Available options relating to this Transform."""
        group = parser.add_argument_group("Transform/Non-zero numerals filter")
        group.add("--nonzero_threshold", "-nonzero_threshold", type=float, default=0.5,
                  help="Threshold for discarding sentences based on numerals between the segments with zeros removed")

    def _parse_opts(self):
        self.nonzero_threshold = self.opts.nonzero_threshold

    def apply(self, example, **kwargs):
        """Return None if the penalty is smaller than the threshold."""
        src = ' '.join(example['src'])
        tgt = ' '.join(example['tgt'])
        nums = [[int(c) for c in sent if c in string.digits and c != '0'] for sent in [src, tgt]]
        for num1, num2 in itertools.combinations(nums, 2):
            seq = difflib.SequenceMatcher(None, num1, num2)
            ratio = seq.ratio()
        if ratio >= self.nonzero_threshold:
            return example
        else:
            return None

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('nonzero_threshold', self.nonzero_threshold)
