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
