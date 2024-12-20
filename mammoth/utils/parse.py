import configargparse as cfargparse
import os
import re
import torch
import yaml

import mammoth.opts as opts
from mammoth.utils.logging import logger
from mammoth.constants import CorpusName, ModelTask
from mammoth.transforms import AVAILABLE_TRANSFORMS

RE_NODE_GPU = re.compile(r'\d+:\d+')
RE_SRC_TGT = re.compile(r'[^-]+-[^-]+')


def yaml_or_dict(val, name):
    if isinstance(val, str):
        return yaml.safe_load(val)
    elif isinstance(val, dict):
        return val
    else:
        raise TypeError(f'{name} {type(val)}')


class DataOptsCheckerMixin(object):
    """Checker with methods for validate data related options."""

    @staticmethod
    def _validate_file(file_path, info):
        """Check `file_path` is valid or raise `IOError`."""
        if not os.path.isfile(file_path):
            raise IOError(f"Please check path of your {info} file! {file_path}")

    @classmethod
    def _validate_adapters(cls, opts):
        """Parse corpora specified in data field of YAML file."""
        if not opts.adapters:
            return
        adapter_opts = yaml_or_dict(opts.adapters, name='opts.adapters')
        # TODO: validate adapter opts
        opts.adapters = adapter_opts

    @classmethod
    def _validate_tasks(cls, opts):
        """Parse tasks/language-pairs/corpora specified in data field of YAML file."""
        default_transforms = opts.transforms
        if len(default_transforms) != 0:
            logger.info(f"Default transforms: {default_transforms}.")
        corpora = yaml_or_dict(opts.tasks, name='opts.tasks')
        logger.info("Parsing corpora")
        n_without_node_gpu = 0
        for cname, corpus in corpora.items():
            logger.info("Parsing corpus '{}': {}".format(cname, corpus))
            # Check Transforms
            _transforms = corpus.get('transforms', None)
            if _transforms is None:
                logger.info(f"Missing transforms field for {cname} data, set to default: {default_transforms}.")
                corpus['transforms'] = default_transforms
            opts.data_task = ModelTask.SEQ2SEQ
            """
            # Check path
            path_src = corpus.get('path_src', None)
            path_tgt = corpus.get('path_tgt', None)
            if path_src is None:
                raise ValueError(f'Corpus {cname} src path is required.'
                                 'tgt path is also required for non language'
                                 ' modeling tasks.')
            else:
                opts.data_task = ModelTask.SEQ2SEQ
                if path_tgt is None:
                    logger.warning(
                        "path_tgt is None, it should be set unless the task"
                        " is language modeling"
                    )
                    opts.data_task = ModelTask.LANGUAGE_MODEL
                    # tgt is src for LM task
                    corpus["path_tgt"] = path_src
                    corpora[cname] = corpus
                    path_tgt = path_src
                cls._validate_file(path_src, info=f'{cname}/path_src')
                cls._validate_file(path_tgt, info=f'{cname}/path_tgt')
            """
            # Check prefix: will be used when use prefix transform
            src_prefix = corpus.get('src_prefix', None)
            tgt_prefix = corpus.get('tgt_prefix', None)
            if src_prefix is None or tgt_prefix is None:
                if 'prefix' in corpus['transforms']:
                    raise ValueError(f'Corpus {cname} prefix are required.')
            # Check weight
            weight = corpus.get('weight', None)
            if weight is None:
                if cname != CorpusName.VALID:
                    logger.warning(f"Corpus {cname}'s weight should be given. We default it to 1 for you.")
                corpus['weight'] = 1
            # Check curriculum introduce_at_training_step
            introduce_at_training_step = corpus.get('introduce_at_training_step', None)
            if introduce_at_training_step is None:
                if cname != CorpusName.VALID:
                    logger.warning(
                        f"Corpus {cname}'s introduce_at_training_step is unset. "
                        " (curriculum introduces the corpus at this step)"
                        " We default it to 0 (start of training) for you."
                    )
                corpus['introduce_at_training_step'] = 0
            enable_embeddingless = corpus.get('enable_embeddingless', False)
            opts.enable_embeddingless = enable_embeddingless
            # Check sharing groups
            enc_sharing_group = corpus.get('enc_sharing_group', None)
            assert enc_sharing_group is None or isinstance(enc_sharing_group, list)
            dec_sharing_group = corpus.get('dec_sharing_group', None)
            assert dec_sharing_group is None or isinstance(dec_sharing_group, list)
            # Node and gpu assignments
            node_gpu = corpus.get('node_gpu', None)
            if node_gpu is not None:
                assert RE_NODE_GPU.match(node_gpu)
            else:
                n_without_node_gpu += 1
            src_tgt = corpus.get('src_tgt', None)
            assert src_tgt is not None
            assert RE_SRC_TGT.match(src_tgt)

            # Check features
            src_feats = corpus.get("src_feats", None)
            if src_feats is not None:
                for feature_name, feature_file in src_feats.items():
                    cls._validate_file(feature_file, info=f'{cname}/path_{feature_name}')
                if 'inferfeats' not in corpus["transforms"]:
                    raise ValueError("'inferfeats' transform is required when setting source features")
                if 'filterfeats' not in corpus["transforms"]:
                    raise ValueError("'filterfeats' transform is required when setting source features")
            else:
                corpus["src_feats"] = None

            stride = corpus.get("stride", None)
            offset = corpus.get("offset", None)
            if stride is not None or offset is not None:
                if stride is None:
                    raise ValueError('stride and offset must be used together')
                if offset is None:
                    raise ValueError('stride and offset must be used together')
                if offset > stride:
                    logger.warning(f'offset {offset} stride {stride} is probably not what you want')

        # Either all tasks should be assigned to a gpu, or none
        assert n_without_node_gpu == 0 or n_without_node_gpu == len(corpora), \
            f'n_without_node_gpu: {n_without_node_gpu} not in {{ 0, {len(corpora)} }}'

        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opts.tasks = corpora

        src_vocab = yaml_or_dict(opts.src_vocab, name="opts.src_vocab")
        logger.info(f"Parsed {len(src_vocab)} vocabs from -src_vocab.")
        opts.src_vocab = src_vocab

        tgt_vocab = yaml_or_dict(opts.tgt_vocab, name="opts.tgt_vocab")
        logger.info(f"Parsed {len(tgt_vocab)} vocabs from -tgt_vocab.")
        opts.tgt_vocab = tgt_vocab

    @classmethod
    def _validate_transforms_opts(cls, opts):
        """Check options used by transforms."""
        for name, transform_cls in AVAILABLE_TRANSFORMS.items():
            if name in opts._all_transform:
                transform_cls._validate_options(opts)

    @classmethod
    def _get_all_transform(cls, opts):
        """Should only called after `_validate_tasks`."""
        all_transforms = set(opts.transforms)
        for cname, corpus in opts.tasks.items():
            _transforms = set(corpus['transforms'])
            if len(_transforms) != 0:
                all_transforms.update(_transforms)
        opts._all_transform = all_transforms

    @classmethod
    def _get_all_transform_translate(cls, opts):
        opts._all_transform = opts.transforms

    @classmethod
    def _validate_fields_opts(cls, opts):
        """Check options relate to vocab and fields."""

        for cname, corpus in opts.tasks.items():
            if cname != CorpusName.VALID and corpus["src_feats"] is not None:
                assert opts.src_feats_vocab, "-src_feats_vocab is required if using source features."
                if isinstance(opts.src_feats_vocab, str):
                    opts.src_feats_vocab = yaml_or_dict(opts.src_feats_vocab, name="opts.src_feats_vocab")

                for feature in corpus["src_feats"].keys():
                    assert feature in opts.src_feats_vocab, f"No vocab file set for feature {feature}"

        # validation when train:
        for key, vocab in opts.src_vocab.items():
            cls._validate_file(vocab, info=f'src vocab ({key})')
            cls._validate_file(vocab, info=f'tgt vocab ({key})')

    @classmethod
    def _validate_language_model_compatibilities_opts(cls, opts):
        if opts.model_task != ModelTask.LANGUAGE_MODEL:
            return

        logger.info("encoder is not used for LM task")

        assert opts.tgt_vocab is None, "vocab must be shared for LM task"

    @classmethod
    def validate_prepare_opts(cls, opts):
        """Validate all options relate to prepare (data/transform/vocab)."""
        cls._validate_tasks(opts)
        cls._get_all_transform(opts)
        cls._validate_transforms_opts(opts)
        cls._validate_fields_opts(opts)

    @classmethod
    def validate_model_opts(cls, opts):
        cls._validate_language_model_compatibilities_opts(opts)


class ArgumentParser(cfargparse.ArgumentParser, DataOptsCheckerMixin):
    """OpenNMT option parser powered with option check methods."""

    def __init__(
        self,
        config_file_parser_class=cfargparse.YAMLConfigFileParser,
        formatter_class=cfargparse.ArgumentDefaultsHelpFormatter,
        **kwargs,
    ):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class, formatter_class=formatter_class, **kwargs
        )
        self.translation = False

    @classmethod
    def defaults(cls, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls()
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    def parse_known_args(self, *args, strict=True, **kwargs):
        opts, unknown = super().parse_known_args(*args, **kwargs)
        strict = strict and not self.translation
        if strict and unknown:
            raise ValueError(f'unknown arguments provided:\n{unknown}')
        if self.translation:
            unknown = []
        return opts, unknown

    @classmethod
    def update_model_opts(cls, model_opts):
        cls._validate_adapters(model_opts)
        if model_opts.model_dim > 0:
            model_opts.model_dim = model_opts.model_dim
            model_opts.model_dim = model_opts.model_dim

        # Backward compatibility with "fix_word_vecs_*" opts
        if hasattr(model_opts, 'fix_word_vecs_enc'):
            model_opts.freeze_word_vecs_enc = model_opts.fix_word_vecs_enc
        if hasattr(model_opts, 'fix_word_vecs_dec'):
            model_opts.freeze_word_vecs_dec = model_opts.fix_word_vecs_dec

    @classmethod
    def validate_x_transformers_opts(cls, opts):
        if not opts.x_transformers_opts:
            opts.x_transformers_opts = dict()
            return
        opts_dict = yaml_or_dict(opts.x_transformers_opts, name="opts.x_transformers_opts")
        for overwritten_key in (
            'dim',
            'depth',
            'causal',
            'cross_attend',
            'pre_norm_has_final_norm',
        ):
            if overwritten_key in opts_dict:
                raise ValueError(
                    f'"{overwritten_key}" is overwritten with values from other Mammoth arguments. '
                    'You can not set it as part of x_transformers_opts.'
                )
        for unsupported_key in (
            'sandwich_coef',    # Sandwich would be very unintuitive with multiple layerstacks
            'macaron',          # Can not support macaron while injecting adapters at each 'f' layer
        ):
            if unsupported_key in opts_dict:
                raise ValueError(
                    f'"{unsupported_key}" is not supported in Mammoth.'
                )
        # Mammoth has a different default value than x-transformers,
        # but you can set these explicitly
        if 'use_simple_rmsnorm' not in opts_dict:
            opts_dict['use_simple_rmsnorm'] = True
        if 'attn_flash' not in opts_dict:
            opts_dict['attn_flash'] = True
        if 'ff_glu' not in opts_dict:
            opts_dict['ff_glu'] = True

        opts.x_transformers_opts = opts_dict

    @classmethod
    def validate_model_opts(cls, model_opts):
        # assert model_opts.model_type in ["text"], "Unsupported model type %s" % model_opts.model_type

        # encoder and decoder should be same sizes
        # assert same_size, "The encoder and decoder rnns must be the same size for now"

        # if model_opts.share_embeddings:
        #    if model_opts.model_type != "text":
        #        raise AssertionError("--share_embeddings requires --model_type text.")

        cls.validate_x_transformers_opts(model_opts)

    @classmethod
    def checkpoint_model_opts(cls, checkpoint_opt):
        # Load default opts values, then overwrite with the opts in
        # the checkpoint. That way, if there are new options added,
        # the defaults are used.
        the_opts = cls.defaults(opts.model_opts)
        the_opts.__dict__.update(checkpoint_opt.__dict__)
        return the_opts

    @classmethod
    def validate_train_opts(cls, opts):
        if opts.epochs:
            raise AssertionError("-epochs is deprecated please use -train_steps.")

        if torch.cuda.is_available() and not opts.gpu_ranks:
            logger.warn("You have a CUDA device, should run with -gpu_ranks")
        if opts.world_size < len(opts.gpu_ranks):
            raise AssertionError("parameter counts of -gpu_ranks must be less or equal than -world_size.")
        if len(opts.gpu_ranks) > 0 and opts.world_size == len(opts.gpu_ranks) and min(opts.gpu_ranks) > 0:
            raise AssertionError(
                "-gpu_ranks should have master(=0) rank unless -world_size is greater than len(gpu_ranks)."
            )

        assert len(opts.dropout) == len(opts.dropout_steps), "Number of dropout values must match accum_steps values"

        assert len(opts.attention_dropout) == len(
            opts.dropout_steps
        ), "Number of attention_dropout values must match accum_steps values"

        assert len(opts.accum_count) == len(
            opts.accum_steps
        ), 'Number of accum_count values must match number of accum_steps'

        if opts.decay_method not in {'linear_warmup', 'none'}:
            logger.warn(
                'Note that decay methods other than "linear_warmup" and "none" have weird scaling.'
                ' Did you tune the learning rate for this decay method?'
            )

        # TODO: do we want to remove that completely?
        # if opts.update_vocab:
        #    assert opts.train_from, "-update_vocab needs -train_from option"
        #    assert opts.reset_optim in ['states', 'all'], '-update_vocab needs -reset_optim "states" or "all"'

    @classmethod
    def validate_translate_opts(cls, opts):
        opts.src_feats = eval(opts.src_feats) if opts.src_feats else {}

    @classmethod
    def validate_translate_opts_dynamic(cls, opts):
        # It comes from training
        # TODO: needs to be added as inference opts
        pass
