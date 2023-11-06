from mammoth.utils.logging import logger
from mammoth.transforms import register_transform
from .transform import Transform


@register_transform(name='prefix')
class PrefixTransform(Transform):
    """Add Prefix to src (& tgt) sentence."""

    def __init__(self, opts):
        super().__init__(opts)

    @staticmethod
    def _get_prefix(corpus):
        """Get prefix string of a `corpus`."""
        if 'prefix' in corpus['transforms']:
            prefix = {'src': corpus['src_prefix'], 'tgt': corpus['tgt_prefix']}
        else:
            prefix = None
        return prefix

    @classmethod
    def get_prefix_dict(cls, opts):
        """Get all needed prefix correspond to corpus in `opts`."""
        prefix_dict = {}
        for c_name, corpus in opts.tasks.items():
            prefix = cls._get_prefix(corpus)
            if prefix is not None:
                logger.info(f"Get prefix for {c_name}: {prefix}")
                prefix_dict[c_name] = prefix
        return prefix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        prefix_dict = cls.get_prefix_dict(opts)
        src_specials, tgt_specials = set(), set()
        for _, prefix in prefix_dict.items():
            src_specials.update(prefix['src'].split())
            tgt_specials.update(prefix['tgt'].split())
        return (src_specials, tgt_specials)

    def warm_up(self, vocabs=None):
        """Warm up to get prefix dictionary."""
        super().warm_up(None)
        self.prefix_dict = self.get_prefix_dict(self.opts)

    def _prepend(self, example, prefix):
        """Prepend `prefix` to `tokens`."""
        for side, side_prefix in prefix.items():
            if example[side] is not None:
                example[side] = side_prefix.split() + example[side]
        return example

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply prefix prepend to example.

        Should provide `corpus_name` to get corresponding prefix.
        """
        corpus_name = kwargs.get('corpus_name', None)
        if corpus_name is None:
            raise ValueError('corpus_name is required.')
        corpus_prefix = self.prefix_dict.get(corpus_name, None)
        if corpus_prefix is None:
            raise ValueError(f'prefix for {corpus_name} does not exist.')
        return self._prepend(example, corpus_prefix)

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('prefix_dict', self.prefix_dict)
