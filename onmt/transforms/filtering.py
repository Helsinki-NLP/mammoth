from onmt.transforms import register_transform
from .transform import Transform, ObservableStats

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
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--src_seq_length", "-src_seq_length", type=int, default=200, help="Maximum source sequence length.")
        group.add("--tgt_seq_length", "-tgt_seq_length", type=int, default=200, help="Maximum target sequence length.")

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length

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
            return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format('src_seq_length', self.src_seq_length, 'tgt_seq_length', self.tgt_seq_length)


@register_transform(name='filtertoolong_opusfilter')
class FilterTooLongTransformOP(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)
        import opusfilter as op

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--src_seq_length_op", "-src_seq_length_op", type=int, default=200, help="Maximum source sequence length.")
        group.add("--tgt_seq_length_op", "-tgt_seq_length_op", type=int, default=200, help="Maximum target sequence length.")

    def _parse_opts(self):
        self.src_seq_length = self.opts.src_seq_length
        self.tgt_seq_length = self.opts.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Return None if too long else return as is."""
        lf_src = op.filters.LengthFilter(max_length=self.src_seq_length)
        length_src = lf_src.get_length(example['src'], 0)
        lf_tgt = op.filters.LengthFilter(max_length=self.src_seq_length)
        length_tgt = lf_tgt.get_length(example['tgt'], 0)
        if lf_src.accept([length_src]) and lf_tgt.accept([length_tgt]):
            return example
        else:
            return None

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format('src_seq_length_op', self.src_seq_length, 'tgt_seq_length_op', self.tgt_seq_length)


# Filters inspired by OpusFilter https://github.com/Helsinki-NLP/OpusFilter/blob/aca40bd064d9b087c5216de0568d7fb91a31d142/opusfilter/filters.py

@register_transform(name='filterwordratio')
class FilterWordRatio(Transform):
    """Filter out sentence based on word length ratio"""

    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--word_ratio_threshold", "-word_ratio_threshold", type=int, default=3, help="Threshold for discarding sentences based on word ratio.")

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
    
@register_transform(name='filterwordratio_opusfilter')
class FilterWordRatioOP(Transform):
    """Filter out sentence based on word length ratio"""
    
    def __init__(self, opts):
        super().__init__(opts)

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Filter")
        group.add("--word_ratio_threshold_op", "-word_ratio_threshold_op", type=int, default=3, help="Threshold for discarding sentences based on word ratio.")
        group.add("--word_ratio_unit_op", "-word_ratio_unit_op", type=str, default="word", choices=[('word', 'char', 'character')], help="Unit for discarding sentences based on char/character/word ratio.")

    def _parse_opts(self):
        self.word_ratio_threshold = self.opts.word_ratio_threshold
        self.word_ratio_unit = self.opts.word_ratio_unit

    def apply(self, example, **kwargs):
        ratiofilter = op.filters.LengthRatioFilter(threshold=self.opts.word_ratio_threshold, unit=self.opts.word_ratio_unit)
        score = ratiofilter.score([example['src'],example['tgt']])
        accept = ratiofilter.accept(next(score))
        if accept:
            return example
        else:
            return None
    
    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('word_ratio_threshold_op', self.word_ratio_threshold)