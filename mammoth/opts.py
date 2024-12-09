""" Implementation of all available options """
import configargparse

from mammoth.modules.position_ffn import ACTIVATION_FUNCTIONS
from mammoth.modules.position_ffn import ActivationFunction
from mammoth.transforms import AVAILABLE_TRANSFORMS
from mammoth.distributed import TASK_DISTRIBUTION_STRATEGIES


def config_opts(parser):
    group = parser.add_argument_group("Configuration")
    group.add('-config', '--config', required=False, is_config_file_arg=True, help='Path of the main YAML config file.')
    group.add(
        '-save_config',
        '--save_config',
        required=False,
        is_write_out_config_file_arg=True,
        help='Path where to save the config.',
    )


def _add_logging_opts(parser, is_train=True):
    group = parser.add_argument_group('Logging')
    group.add('--log_file', '-log_file', type=str, default="", help="Output logs to a file under this path.")
    group.add(
        '--structured_log_file',
        '-structured_log_file',
        type=str,
        default="",
        help="Output machine-readable structured logs to a file under this path."
    )
    group.add(
        '--log_file_level',
        '-log_file_level',
        type=str,
        action=StoreLoggingLevelAction,
        choices=StoreLoggingLevelAction.CHOICES,
        default="0",
    )
    group.add(
        '--verbose',
        '-verbose',
        action="store_true",
        help='Print data loading and statistics for all process (default only log the first process shard)'
        if is_train
        else 'Print scores and predictions for each sentence',
    )
    group.add(
        '--log_model_structure',
        '-log_model_structure',
        action="store_true",
        help='Print the entire model structure when building the model. Verbose, but useful for debugging.'
    )

    if is_train:
        group.add('--report_every', '-report_every', type=int, default=50, help="Print stats at this interval.")
        group.add('--exp_host', '-exp_host', type=str, default="", help="Send logs to this crayon server.")
        group.add('--exp', '-exp', type=str, default="", help="Name of the experiment for logging.")
        # Use Tensorboard for visualization during training
        group.add(
            '--tensorboard',
            '-tensorboard',
            action="store_true",
            help="Use tensorboard for visualization during training. Must have the library tensorboard >= 1.14.",
        )
        group.add(
            "--tensorboard_log_dir",
            "-tensorboard_log_dir",
            type=str,
            default="runs/mammoth",
            help="Log directory for Tensorboard. This is also the name of the run.",
        )
        group.add(
            '--report_stats_from_parameters',
            '-report_stats_from_parameters',
            action="store_true",
            help="Report parameter-level statistics in tensorboard. "
            "This has a huge impact on performance: only use for debugging.",
        )
        group.add(
            '--report_training_accuracy',
            '-report_trainig_accuracy',
            action="store_true",
            help="Report accuracy for training batches. "
            "This has an impact on performance: only use for debugging, or if no validation set exists.",
        )

    else:
        # Options only during inference
        group.add('--attn_debug', '-attn_debug', action="store_true", help='Print best attn for each word')
        group.add('--align_debug', '-align_debug', action="store_true", help='Print best align for each word')
        group.add('--dump_beam', '-dump_beam', type=str, default="", help='File to dump beam information to.')
        group.add(
            '--n_best',
            '-n_best',
            type=int,
            default=1,
            help="If verbose is set, will output the n_best decoded sentences",
        )


def _add_reproducibility_opts(parser):
    group = parser.add_argument_group('Reproducibility')
    group.add(
        '--seed',
        '-seed',
        type=int,
        required=True,
        help="Set random seed used for better reproducibility between experiments. "
        "Mandatory for multi-gpu training, and for convenience required for all.",
    )


def _add_dynamic_corpus_opts(parser):
    """Options related to training corpus, type: a list of dictionary."""
    group = parser.add_argument_group('Data/Tasks')
    group.add(
        "-tasks",
        "--tasks",
        required=True,
        help="List of datasets and their specifications. See examples/*.yaml for further details.",
    )
    group.add(
        "-transforms",
        "--transforms",
        default=[],
        nargs="+",
        choices=AVAILABLE_TRANSFORMS.keys(),
        help="Default transform pipeline to apply to data. Can be specified in each corpus of data to override.",
    )


def _add_dynamic_vocabs_opts(parser):
    """Options related to vocabulary.

    Add all options relate to vocabulary to parser.
    """
    group = parser.add_argument_group("Vocab")
    group.add(
        "-src_vocab",
        "--src_vocab",
        required=True,
        help="Path to src (or shared) vocabulary file. "
        "Format: one <word> or <word>\t<count> per line.",
    )
    group.add(
        "-tgt_vocab",
        "--tgt_vocab",
        help="Path to tgt vocabulary file. "
        "Format: one <word> or <word>\t<count> per line.",
    )

    group.add(
        "-src_vocab_size",
        "--src_vocab_size",
        type=int,
        default=None,
        help="Maximum size of the source vocabulary; will silently truncate your vocab file if longer.",
    )
    group.add(
        "-tgt_vocab_size",
        "--tgt_vocab_size",
        type=int,
        default=None,
        help="Maximum size of the target vocabulary; will silently truncate your vocab file if longer."
    )


def _add_dynamic_transform_opts(parser):
    """Options related to transforms.

    Options that specified in the definitions of each transform class
    at `mammoth/transforms/*.py`.
    """
    for name, transform_cls in AVAILABLE_TRANSFORMS.items():
        transform_cls.add_options(parser)


def dynamic_prepare_opts(parser):
    """Options related to data prepare in dynamic mode.

    Add all dynamic data prepare related options to parser.
    """
    config_opts(parser)
    _add_dynamic_corpus_opts(parser)
    _add_dynamic_vocabs_opts(parser)
    _add_dynamic_transform_opts(parser)


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model- Embeddings')

    group.add(
        '--enable_embeddingless',
        '-enable_embeddingless',
        action='store_true',
        help="Enable the use of byte-based embeddingless models" +
        "(Shaham et. al, 2021) https://aclanthology.org/2021.naacl-main.17/",
    )

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add(
        '--model_type',
        '-model_type',
        default='text',
        choices=['text'],
        help="Type of source model to use. Allows the system to incorporate non-text inputs. Options are [text].",
    )
    group.add('--model_dtype', '-model_dtype', default='fp32', choices=['fp32', 'fp16'], help='Data type of the model.')

    # group.add('--layers', '-layers', type=int, default=-1, help='Deprecated')
    group.add('--enc_layers', '-enc_layers', nargs='+', type=int, help='Number of layers in each encoder module')
    group.add('--dec_layers', '-dec_layers', nargs='+', type=int, help='Number of layers in each decoder module')
    group.add(
        '--model_dim',
        '-model_dim',
        type=int,
        default=-1,
        help="Size of Transformer representations.",
    )

    group.add(
        '--pos_ffn_activation_fn',
        '-pos_ffn_activation_fn',
        type=str,
        default=ActivationFunction.relu,
        choices=ACTIVATION_FUNCTIONS.keys(),
        help='The activation'
        ' function to use in PositionwiseFeedForward layer. Choices are'
        f' {ACTIVATION_FUNCTIONS.keys()}. Default to'
        f' {ActivationFunction.relu}.',
    )

    group.add('-normformer', '--normformer', action='store_true', help='NormFormer-style normalization')

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add(
        '--self_attn_type',
        '-self_attn_type',
        type=str,
        default="scaled-dot",
        help='Self attention type in Transformer decoder layer -- currently "scaled-dot" or "average" ',
    )

    # TODO is this actually in use?
    group.add(
        '--max_relative_positions',
        '-max_relative_positions',
        type=int,
        default=0,
        help="Maximum distance between inputs in relative "
        "positions representations. "
        "For more detailed information, see: "
        "https://arxiv.org/pdf/1803.02155.pdf",
    )
    group.add(
        "-x_transformers_opts",
        "--x_transformers_opts",
        help="For a complete list of options (name only), see the code"
        " https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py ."
        " The kwargs of `AttentionLayers` can be used without prefix,"
        " the kwargs of `FeedForward` with the prefix `ff_`,"
        " and the kwargs of `Attention` with the prefix `attn_`."
        " For tips, examples, and citations see"
        " https://github.com/lucidrains/x-transformers/blob/main/README.md ."
    )

    # Generator and loss options.
    group = parser.add_argument_group('Generator')
    group.add(
        '--generator_function',
        '-generator_function',
        default="softmax",
        choices=["softmax"],
        help="Which function to use for generating "
        "probabilities over the target vocabulary (choices: "
        "softmax)",
    )
    group.add(
        '--loss_scale',
        '-loss_scale',
        type=float,
        default=0.0,
        help="For FP16 training, the static loss scale to use. If not set, the loss scale is dynamically computed.",
    )
    group.add(
        '--apex_opt_level',
        '-apex_opt_level',
        type=str,
        default="O1",
        choices=["O0", "O1", "O2", "O3"],
        help="For FP16 training, the opt_level to use. See https://nvidia.github.io/apex/amp.html#opts-levels.",
    )

    # attention bridge options
    group = parser.add_argument_group("Attention bridge")
    group.add(
        '--hidden_ab_size', '-hidden_ab_size', type=int, default=2048, help="""Size of attention bridge hidden states"""
    )
    group.add(
        '--ab_fixed_length',
        '-ab_fixed_length',
        type=int,
        default=50,
        help="Number of attention heads in attention bridge (fixed length of output)",
    )
    group.add(
        '--ab_layers',
        '-ab_layers',
        nargs='*',
        default=[],
        choices=['lin', 'simple', 'transformer', 'perceiver', 'feedforward'],
        help="Composition of the attention bridge",
    )
    group.add(
        '--ab_layer_norm',
        '-ab_layer_norm',
        type=str,
        default='layernorm',
        choices=['none', 'rmsnorm', 'layernorm'],
        help="""Use layer normalization after lin, simple and feedforward bridge layers""",
    )
    group.add(
        '--ab_heads', '-ab_heads', type=int, default=8,
        help='Number of heads for transformer self-attention. '
        ' Semi-obsolete: not used for x-transformers, only used for some attention bridge configuations.'
    )
    group.add(
        '--dropout',
        '-dropout',
        type=float,
        default=[0.3],
        nargs='+',
        help="Dropout probability; Legacy: applied in the attention bridge",
    )
    group.add(
        '--attention_dropout',
        '-attention_dropout',
        type=float,
        default=[0.1],
        nargs='+',
        help="Attention Dropout probability; Legacy: applied in the attention bridge",
    )

    # adapter options are in a dict "adapters", and in the corpus options
    group = parser.add_argument_group("Adapters")
    group.add('-adapters', '--adapters',
              help="""Adapter specifications""")


def _add_train_general_opts(parser):
    """General options for training"""
    group = parser.add_argument_group('General')
    # TODO maybe relevant for issue #53
    # group.add('--data_type', '-data_type', default="text", help="Type of the source input. Options are [text].")

    group.add(
        '--save_model',
        '-save_model',
        default='model',
        help="Model filename (the model will be saved as <save_model>_N.pt where N is the number of steps",
    )

    group.add(
        '--save_checkpoint_steps',
        '-save_checkpoint_steps',
        type=int,
        default=5000,
        help="""Save a checkpoint every X steps""",
    )
    group.add(
        '--keep_checkpoint', '-keep_checkpoint', type=int, default=-1, help="Keep X checkpoints (negative: keep all)"
    )
    group.add('--train_steps', '-train_steps', type=int, default=100000, help='Number of training steps')
    group.add('--epochs', '-epochs', type=int, default=0, help='Deprecated epochs see train_steps')
    group.add('--valid_steps', '-valid_steps', type=int, default=10000, help='Perfom validation every X steps')
    group.add(
        '--early_stopping', '-early_stopping', type=int, default=0, help='Number of validation steps without improving.'
    )
    group.add(
        '--early_stopping_criteria',
        '-early_stopping_criteria',
        nargs="*",
        default=None,
        help='Criteria to use for early stopping.',
    )

    # GPU
    group = parser.add_argument_group('Computation Environment')
    group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int, help="list of ranks of each process.")
    group.add('--n_nodes', '-n_nodes', default=1, type=int, help="total number of training nodes.")
    group.add(
        '--node_rank',
        '-node_rank',
        required=True,
        type=int,
        help="index of current node (0-based). "
             "When using non-distributed training (CPU, single-GPU), set to 0"
    )
    group.add('--world_size', '-world_size', default=1, type=int, help="total number of distributed processes.")
    # TODO is gpu_backend actually in use?
    group.add('--gpu_backend', '-gpu_backend', default="nccl", type=str, help="Type of torch distributed backend")
    group.add(
        '--gpu_verbose_level',
        '-gpu_verbose_level',
        default=0,
        type=int,
        help="Gives more info on each process per GPU.",
    )
    group.add(
        '--master_ip', '-master_ip', default="localhost", type=str, help="IP of master for torch.distributed training."
    )
    group.add(
        '--master_port', '-master_port', default=10000, type=int, help="Port of master for torch.distributed training."
    )
    group.add(
        '--queue_size', '-queue_size', default=40, type=int, help="Size of queue for each process in producer/consumer"
    )

    _add_reproducibility_opts(parser)

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add(
        '--param_init',
        '-param_init',
        type=float,
        default=0.1,
        help="Legacy opt for attention bridge. Parameters are initialized over uniform distribution "
        "with support (-param_init, param_init). "
        "Use 0 to not use initialization",
    )
    group.add(
        '--param_init_glorot',
        '-param_init_glorot',
        action='store_true',
        help="Legacy opt for attention bridge. Init parameters with xavier_uniform.",
    )

    group.add(
        '--train_from',
        '-train_from',
        default='',
        type=str,
        help="If training from a checkpoint then this is the path to the pretrained model's state_dict.",
    )
    group.add(
        '--reset_optim',
        '-reset_optim',
        default='none',
        choices=['none', 'all', 'states', 'keep_states'],
        help="Optimization resetter when train_from.",
    )
    group.add(
        '--yes_i_messed_with_the_checkpoint',
        '-yes_i_messed_with_the_checkpoint',
        action='store_true',
        help="Only set this if you know what you are doing."
    )

    # Freeze word vectors
    group.add(
        '--freeze_word_vecs_enc',
        '-freeze_word_vecs_enc',
        action='store_true',
        help="Freeze word embeddings on the encoder side.",
    )
    group.add(
        '--freeze_word_vecs_dec',
        '-freeze_word_vecs_dec',
        action='store_true',
        help="Freeze word embeddings on the decoder side.",
    )

    # Optimization options
    group = parser.add_argument_group('Batching')
    group.add('--batch_size', '-batch_size', type=int, default=64, help='Maximum batch size for training')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32, help='Maximum batch size for validation')
    group.add(
        '--batch_size_multiple',
        '-batch_size_multiple',
        type=int,
        default=None,
        help='Batch size multiple for token batches.',
    )
    group.add(
        '--batch_type',
        '-batch_type',
        default='sents',
        choices=["sents", "tokens"],
        help="Batch grouping for batch_size. Standard is sents. Tokens will do dynamic batching",
    )
    group.add(
        '--pad_to_max_length',
        '-pad_to_max_length',
        action='store_true',
        help='Pad all minibatches to max_length instead of to the length of the longest sequence in the minibatch. '
        'Using this together with batch_type=sents results in tensors of a fixed shape.'
    )
    group.add('--max_length', '-max_length', type=int, default=None, help='Maximum sequence length.')
    group.add(
        '--task_distribution_strategy',
        '-task_distribution_strategy',
        choices=TASK_DISTRIBUTION_STRATEGIES.keys(),
        default='weighted_sampling',
        help="Strategy for the order in which tasks (e.g. language pairs) are scheduled for training"
    )
    group.add(
        "-lookahead_minibatches",
        "--lookahead_minibatches",
        type=int,
        default=4,
        help="The number of minibatches that SimpleLookAheadBucketing will read into a maxibatch, "
        "pessimisticly sort by length, split into minibatches, and yield in one go. "
        "Recommended value: same as accum_count, or at least a multiple of it."
    )
    group.add(
        "-max_look_ahead_sentences",
        "--max_look_ahead_sentences",
        type=int,
        default=2048,
        help="(Maximum) number of sentence pairs that SimpleLookAheadBucketing can attempt to add to the maxibatch. "
        "This is mainly a failsafe in case some corpus contains very short examples.",
    )

    group.add(
        '--optim',
        '-optim',
        default='sgd',
        choices=['sgd', 'adagrad', 'adadelta', 'adam', 'adamw', 'adafactor', 'fusedadam'],
        help="Optimization method.",
    )
    group.add(
        '--adagrad_accumulator_init',
        '-adagrad_accumulator_init',
        type=float,
        default=0.0,
        help="Initializes the accumulator values in adagrad. "
        "Mirrors the initial_accumulator_value option "
        "in the tensorflow adagrad (use 0.1 for their default).",
    )
    group.add(
        '--max_grad_norm',
        '-max_grad_norm',
        type=float,
        default=1.0,
        help="If the norm of the gradient vector exceeds this, "
        "renormalize it to have the norm equal to "
        "max_grad_norm",
    )
    group.add(
        '--weight_decay',
        '-weight_decay',
        type=float,
        default=0.0,
        help="L2 penalty (weight decay) regularizer",
    )
    group.add(
        '--dropout_steps', '-dropout_steps', type=int, nargs='+', default=[0], help="Steps at which dropout changes."
    )
    group.add(
        '--adam_beta1',
        '-adam_beta1',
        type=float,
        default=0.9,
        help="The beta1 parameter used by Adam. "
        "Almost without exception a value of 0.9 is used in "
        "the literature, seemingly giving good results, "
        "so we would discourage changing this value from "
        "the default without due consideration.",
    )
    group.add(
        '--adam_beta2',
        '-adam_beta2',
        type=float,
        default=0.999,
        help='The beta2 parameter used by Adam. '
        'Typically a value of 0.999 is recommended, as this is '
        'the value suggested by the original paper describing '
        'Adam, and is also the value adopted in other frameworks '
        'such as Tensorflow and Keras, i.e. see: '
        'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
        'Optimizer or https://keras.io/optimizers/ . '
        'Whereas recently the paper "Attention is All You Need" '
        'suggested a value of 0.98 for beta2, this parameter may '
        'not work well for normal models / default '
        'baselines.',
    )
    group.add(
        '--label_smoothing',
        '-label_smoothing',
        type=float,
        default=0.0,
        help="Label smoothing value epsilon. "
        "Probabilities of all non-true labels "
        "will be smoothed by epsilon / (vocab_size - 1). "
        "Set to zero to turn off label smoothing. "
        "For more detailed information, see: "
        "https://arxiv.org/abs/1512.00567",
    )
    group.add(
        '--average_decay',
        '-average_decay',
        type=float,
        default=0.0,
        help="Moving average decay. "
        "Set to other than 0 (e.g. 1e-4) to activate. "
        "Similar to Marian NMT implementation: "
        "http://www.aclweb.org/anthology/P18-4020 "
        "For more detail on Exponential Moving Average: "
        "https://en.wikipedia.org/wiki/Moving_average",
    )
    group.add(
        '--average_every',
        '-average_every',
        type=int,
        default=1,
        help="Step for moving average. Default is every update, if -average_decay is set.",
    )
    group.add(
        '--normalization',
        '-normalization',
        default='sents',
        choices=["sents", "tokens"],
        help='Normalization method of the gradient.',
    )
    group.add(
        '--accum_count',
        '-accum_count',
        type=int,
        nargs='+',
        default=[1],
        help="Accumulate gradient this many times. "
        "Approximately equivalent to updating "
        "batch_size * accum_count batches at once. "
        "Recommended for Transformer.",
    )
    group.add(
        '--accum_steps',
        '-accum_steps',
        type=int,
        nargs='+',
        default=[0],
        help="Steps at which accum_count values change",
    )

    group.add(
        '--learning_rate',
        '-learning_rate',
        type=float,
        default=1.0,
        help="Starting learning rate. ",
        # "Recommended settings: sgd = TBD, adagrad = TBD, adadelta = TBD, adam = TBD",
    )
    group.add(
        '--learning_rate_decay',
        '-learning_rate_decay',
        type=float,
        default=0.5,
        help="If update_learning_rate, decay learning rate by "
        "this much if steps have gone past "
        "start_decay_steps",
    )
    group.add(
        '--start_decay_steps',
        '-start_decay_steps',
        type=int,
        default=50000,
        help="Start decaying every decay_steps after start_decay_steps",
    )
    group.add('--decay_steps', '-decay_steps', type=int, default=10000, help="Decay every decay_steps")

    group.add(
        '--decay_method',
        '-decay_method',
        type=str,
        default="none",
        choices=['noam', 'noamwd', 'rsqrt', 'linear_warmup', 'none'],
        help="Use a custom decay rate.",
    )
    group.add(
        '--warmup_steps', '-warmup_steps', type=int, default=4000, help="Number of warmup steps for custom decay."
    )
    _add_logging_opts(parser, is_train=True)


def train_opts(parser):
    """All options used in train."""
    # options relate to data preprare
    dynamic_prepare_opts(parser)
    # options relate to train
    model_opts(parser)
    _add_train_general_opts(parser)


def _add_decoding_opts(parser):
    group = parser.add_argument_group('Beam Search')
    beam_size = group.add('--beam_size', '-beam_size', type=int, default=5, help='Beam size')
    group.add('--ratio', '-ratio', type=float, default=-0.0, help="Ratio based beam stop condition")

    group = parser.add_argument_group('Random Sampling')
    group.add(
        '--random_sampling_topk',
        '-random_sampling_topk',
        default=0,
        type=int,
        help="Set this to -1 to do random sampling from full "
        "distribution. Set this to value k>1 to do random "
        "sampling restricted to the k most likely next tokens. "
        "Set this to 1 to use argmax.",
    )
    group.add(
        '--random_sampling_topp',
        '-random_sampling_topp',
        default=0.0,
        type=float,
        help="Probability for top-p/nucleus sampling. Restrict tokens"
        " to the most likely until the cumulated probability is"
        " over p. In range [0, 1]."
        " https://arxiv.org/abs/1904.09751",
    )
    group.add(
        '--random_sampling_temp',
        '-random_sampling_temp',
        default=1.0,
        type=float,
        help="If doing random sampling, divide the logits by this before computing softmax during decoding.",
    )
    group._group_actions.append(beam_size)
    _add_reproducibility_opts(parser)

    group = parser.add_argument_group('Penalties', '.. Note:: Coverage Penalty is not available in sampling.')
    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    # Length penalty options
    group.add(
        '--length_penalty',
        '-length_penalty',
        default='none',
        choices=['none', 'wu', 'avg'],
        help="Length Penalty to use.",
    )
    group.add(
        '--alpha',
        '-alpha',
        type=float,
        default=0.0,
        help="Google NMT length penalty parameter (higher = longer generation)",
    )
    # Coverage penalty options
    group.add(
        '--coverage_penalty',
        '-coverage_penalty',
        default='none',
        choices=['none', 'wu', 'summary'],
        help="Coverage Penalty to use. Only available in beam search.",
    )
    group.add('--beta', '-beta', type=float, default=-0.0, help="Coverage penalty parameter")
    group.add(
        '--stepwise_penalty',
        '-stepwise_penalty',
        action='store_true',
        help="Apply coverage penalty at every decoding step. Helpful for summary penalty.",
    )

    group = parser.add_argument_group(
        'Decoding tricks', '.. Tip:: Following options can be used to limit the decoding length or content.'
    )
    # Decoding Length constraint
    group.add('--min_length', '-min_length', type=int, default=0, help='Minimum prediction length')
    group.add('--max_length', '-max_length', type=int, default=100, help='Maximum prediction length.')
    group.add(
        '--max_sent_length', '-max_sent_length', action=DeprecateAction, help="Deprecated, use `-max_length` instead"
    )
    # Decoding content constraint
    group.add(
        '--block_ngram_repeat',
        '-block_ngram_repeat',
        type=int,
        default=0,
        help='Block repetition of ngrams during decoding.',
    )
    group.add(
        '--ignore_when_blocking',
        '-ignore_when_blocking',
        nargs='+',
        type=str,
        default=[],
        help="Ignore these strings when blocking repeats. You want to block sentence delimiters.",
    )
    group.add(
        '--replace_unk',
        '-replace_unk',
        action="store_true",
        help="Replace the generated UNK tokens with the "
        "source token that had highest attention weight. If "
        "phrase_table is provided, it will look up the "
        "identified source token and give the corresponding "
        "target token. If it is not provided (or the identified "
        "source token does not exist in the table), then it "
        "will copy the source token.",
    )
    group.add(
        '--ban_unk_token',
        '-ban_unk_token',
        action="store_true",
        help="Prevent unk token generation by setting unk proba to 0",
    )
    group.add(
        '--phrase_table',
        '-phrase_table',
        type=str,
        default="",
        help="If phrase_table is provided (with replace_unk), it will "
        "look up the identified source token and give the "
        "corresponding target token. If it is not provided "
        "(or the identified source token does not exist in "
        "the table), then it will copy the source token.",
    )


def translate_opts(parser, dynamic=False):
    """Translation / inference options"""
    group = parser.add_argument_group('Model')
    group.add(
        '--model',
        '-model',
        dest='models',
        metavar='MODEL',
        nargs='+',
        type=str,
        default=[],
        required=True,
        help="Path to model .pt file(s). Multiple models can be specified, for ensemble decoding.",
    )
    group.add(
        '--fp32',
        '-fp32',
        action='store_true',
        help="Force the model to be in FP32 because FP16 is very slow on GTX1080(ti).",
    )
    group.add('--int8', '-int8', action='store_true', help="Enable dynamic 8-bit quantization (CPU only).")
    group.add(
        '--avg_raw_probs',
        '-avg_raw_probs',
        action='store_true',
        help="If this is set, during ensembling scores from "
        "different models will be combined by averaging their "
        "raw probabilities and then taking the log. Otherwise, "
        "the log probabilities will be averaged directly. "
        "Necessary for models whose output layers can assign "
        "zero probability.",
    )
    group.add('--task_id', '-task_id', help="Task id to determine components to load for translation", required=True)

    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text", help="Type of the source input. Options: [text].")

    group.add('--src', '-src', required=True, help="Source sequence to decode (one line per sequence)")
    group.add(
        "-src_feats",
        "--src_feats",
        required=False,
        help="Source sequence features (dict format). "
        "Ex: {'feat_0': '../data.txt.feats0', 'feat_1': '../data.txt.feats1'}",
    )  # noqa: E501
    group.add('--tgt', '-tgt', help='True target sequence (optional)')
    group.add(
        '--shard_size',
        '-shard_size',
        type=int,
        default=10000,
        help="Divide src and tgt (if applicable) into "
        "smaller multiple src and tgt files, then "
        "build shards, each shard will have "
        "opts.shard_size samples except last shard. "
        "shard_size=0 means no segmentation "
        "shard_size>0 means segment dataset into multiple shards, "
        "each shard has shard_size samples",
    )
    group.add(
        '--output',
        '-output',
        default='pred.txt',
        help="Path to output the predictions (each line will be the decoded sequence",
    )
    group.add('--report_align', '-report_align', action='store_true', help="Report alignment for each translation.")
    group.add('--report_time', '-report_time', action='store_true', help="Report some translation time metrics")

    # Adding options relate to decoding strategy
    _add_decoding_opts(parser)

    # Adding option for logging
    _add_logging_opts(parser, is_train=False)

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=200, help='Batch size')
    group.add(
        '--batch_type',
        '-batch_type',
        default='tokens',
        choices=["sents", "tokens"],
        help="Batch grouping for batch_size. Standard is tokens (max of src and tgt). Sents is unimplemented.",
    )
    group.add('--gpu_rank', '-gpu_rank', type=int, default=-1, help="Device to run on")

    group.add(
        "--output_model",
        "-output_model",
        required=False,
        help="Path to the model output",
    )


# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


class StoreLoggingLevelAction(configargparse.Action):
    """Convert string to logging level"""

    import logging

    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """Deprecate action"""

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0, help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)
