"""Tests for component-based (zero-shot) inference.

Tests the ability to activate encoder/decoder by component IDs and language
rather than by task ID, enabling zero-shot inference for unseen language pairs.
"""
import pytest
import torch
from torch import nn
from unittest.mock import MagicMock, patch

from mammoth.modules.adapters import AdaptedAttentionLayers
from mammoth.modules.layer_stack import AdaptedAttentionLayersStack, StackXcoder
from mammoth.x_transformers import TransformerWrapper
from mammoth.x_transformers.x_transformers import TokenEmbedding


# ---------------------------------------------------------------------------
# Helpers to build minimal real objects
# ---------------------------------------------------------------------------

DIM = 32
DEPTH = 2
HEADS = 4
VOCAB_SIZE = 64
MAX_SEQ_LEN = 128


def _make_attention_layers(xcoder_id, layer_stack_index=0, causal=False):
    """Build a minimal AdaptedAttentionLayers."""
    return AdaptedAttentionLayers(
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        causal=causal,
        layer_stack_index=layer_stack_index,
        xcoder_id=xcoder_id,
    )


def _make_token_emb(vocab_size=VOCAB_SIZE):
    return TokenEmbedding(dim=DIM, num_tokens=vocab_size)


def _make_transformer_wrapper(attn_layers, token_emb, post_emb_norm=None, pos_emb=None, project_emb=None, to_logits=None):
    """Build a TransformerWrapper from pre-built components."""
    return TransformerWrapper(
        num_tokens=token_emb.emb.num_embeddings,
        max_seq_len=MAX_SEQ_LEN,
        attn_layers=attn_layers,
        emb_dim=DIM,
        token_emb=token_emb,
        post_emb_norm_module=post_emb_norm,
        pos_emb_module=pos_emb,
        project_emb_module=project_emb,
        to_logits=to_logits,
    )


def _build_stack_xcoder_for_training(tasks_config, side='encoder'):
    """
    Build a StackXcoder that simulates a trained model.

    tasks_config: list of dicts, each with keys:
        corpus_id, xcoder_ids (list[str]), lang
    side: 'encoder' or 'decoder'

    Returns the StackXcoder plus the component pieces for assertions.
    """
    causal = (side == 'decoder')

    # Build attention layers by (layer_stack_index, xcoder_id) — shared across tasks
    attention_layer_blocks = {}
    for task_cfg in tasks_config:
        for layer_stack_index, xcoder_id in enumerate(task_cfg['xcoder_ids']):
            if layer_stack_index not in attention_layer_blocks:
                attention_layer_blocks[layer_stack_index] = {}
            if xcoder_id not in attention_layer_blocks[layer_stack_index]:
                attention_layer_blocks[layer_stack_index][xcoder_id] = _make_attention_layers(
                    xcoder_id, layer_stack_index, causal=causal
                )

    # Build token embeddings by lang — shared across tasks
    token_embs = {}
    for task_cfg in tasks_config:
        lang = task_cfg['lang']
        if lang not in token_embs:
            token_embs[lang] = _make_token_emb()

    # Build per-component wrapper modules
    per_component_post_emb_norms = {}
    per_component_pos_embs = {}
    per_component_project_embs = {}
    per_component_to_logits = {}

    for task_cfg in tasks_config:
        component_key = tuple(task_cfg['xcoder_ids'])
        if component_key not in per_component_post_emb_norms:
            per_component_post_emb_norms[component_key] = nn.Identity()
            per_component_pos_embs[component_key] = None
            per_component_project_embs[component_key] = nn.Identity()
            if side == 'decoder':
                per_component_to_logits[component_key] = nn.Linear(DIM, VOCAB_SIZE, bias=False)

    # Build TransformerWrappers per task (the current approach)
    transformer_wrappers = {}
    for task_cfg in tasks_config:
        component_key = tuple(task_cfg['xcoder_ids'])
        attn_layers_list = [
            attention_layer_blocks[i][xid]
            for i, xid in enumerate(task_cfg['xcoder_ids'])
        ]
        attn_layers_stack = AdaptedAttentionLayersStack(attn_layers_list)
        tw = _make_transformer_wrapper(
            attn_layers=attn_layers_stack,
            token_emb=token_embs[task_cfg['lang']],
            post_emb_norm=per_component_post_emb_norms[component_key],
            pos_emb=per_component_pos_embs[component_key],
            project_emb=per_component_project_embs[component_key],
            to_logits=per_component_to_logits.get(component_key),
        )
        transformer_wrappers[task_cfg['corpus_id']] = tw

    stack_xcoder = StackXcoder(
        transformer_wrappers=transformer_wrappers,
        attention_layer_blocks=attention_layer_blocks,
        token_embs=token_embs,
        adapters=None,
        per_component_post_emb_norms=per_component_post_emb_norms,
        per_component_pos_embs=per_component_pos_embs,
        per_component_project_embs=per_component_project_embs,
        per_component_to_logits=per_component_to_logits,
    )
    # Set wrapper kwargs needed by activate_by_components
    stack_xcoder.transformer_wrapper_kwargs = {'max_seq_len': MAX_SEQ_LEN}
    stack_xcoder.emb_dim = DIM
    return stack_xcoder, {
        'attention_layer_blocks': attention_layer_blocks,
        'token_embs': token_embs,
        'per_component_post_emb_norms': per_component_post_emb_norms,
        'per_component_pos_embs': per_component_pos_embs,
        'per_component_project_embs': per_component_project_embs,
        'per_component_to_logits': per_component_to_logits,
    }


# ---------------------------------------------------------------------------
# A typical multilingual training setup:
#   train_en_is: encoder("en") + decoder("is")
#   train_de_is: encoder("de") + decoder("is")
# Zero-shot goal: encoder("de") + decoder("en"), or encoder("en") + decoder("de")
# ---------------------------------------------------------------------------

ENCODER_TASKS = [
    {'corpus_id': 'train_en_is', 'xcoder_ids': ['en'], 'lang': 'en'},
    {'corpus_id': 'train_de_is', 'xcoder_ids': ['de'], 'lang': 'de'},
]

DECODER_TASKS = [
    {'corpus_id': 'train_en_is', 'xcoder_ids': ['is'], 'lang': 'is'},
    {'corpus_id': 'train_de_is', 'xcoder_ids': ['is'], 'lang': 'is'},
]


# ===========================================================================
# Test: StackXcoder.activate_by_components()
# ===========================================================================

class TestActivateByComponents:
    """Tests for the new activate_by_components method on StackXcoder."""

    def test_activate_trained_combination(self):
        """Activating a combination that was seen during training should work
        and return a TransformerWrapper with the correct components."""
        encoder, parts = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        # Activate encoder("en") with lang="en" — this was trained as train_en_is
        tw = encoder.activate_by_components(xcoder_ids=['en'], lang='en')

        assert isinstance(tw, TransformerWrapper)
        # Should use the "en" attention layers
        assert tw.attn_layers.attention_layers_stack[0] is parts['attention_layer_blocks'][0]['en']
        # Should use the "en" token embedding
        assert tw.token_emb is parts['token_embs']['en']

    def test_activate_zero_shot_combination(self):
        """Activating a novel combination (zero-shot) should work:
        use decoder component ("is") but with language embedding "de".

        This simulates having trained en->is and de->is, and now wanting
        to use the "is" decoder with "de" target embeddings (unusual but valid
        for testing). A more realistic zero-shot would be using a decoder
        component with a language embedding that was trained on a different task.
        """
        # For a more realistic test: add a decoder("en") component
        decoder_tasks = [
            {'corpus_id': 'train_en_is', 'xcoder_ids': ['is'], 'lang': 'is'},
            {'corpus_id': 'train_de_is', 'xcoder_ids': ['is'], 'lang': 'is'},
            {'corpus_id': 'train_is_en', 'xcoder_ids': ['en'], 'lang': 'en'},
        ]
        decoder, parts = _build_stack_xcoder_for_training(decoder_tasks, side='decoder')

        # Zero-shot: decoder component "en" (trained for is->en) with lang "is"
        # This combination (component=en, lang=is) was never a training task
        tw = decoder.activate_by_components(xcoder_ids=['en'], lang='is')

        assert isinstance(tw, TransformerWrapper)
        # Attention layers should come from the "en" decoder component
        assert tw.attn_layers.attention_layers_stack[0] is parts['attention_layer_blocks'][0]['en']
        # Token embedding should come from "is" language
        assert tw.token_emb is parts['token_embs']['is']

    def test_activate_by_components_shares_attention_layers_with_activate(self):
        """activate_by_components should use the exact same attention layer
        objects as the task-based activate method."""
        encoder, parts = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        tw_by_task = encoder.activate(task_id='train_en_is', adapter_ids=None)
        tw_by_component = encoder.activate_by_components(xcoder_ids=['en'], lang='en')

        # The underlying attention layers must be the same objects (shared, not copied)
        task_attn = tw_by_task.attn_layers.attention_layers_stack[0]
        comp_attn = tw_by_component.attn_layers.attention_layers_stack[0]
        assert task_attn is comp_attn

    def test_activate_by_components_shares_wrapper_modules(self):
        """Per-component wrapper modules (post_emb_norm, project_emb, etc.)
        should be identical objects between task-based and component-based activation."""
        decoder_tasks = [
            {'corpus_id': 'train_en_is', 'xcoder_ids': ['is'], 'lang': 'is'},
        ]
        decoder, parts = _build_stack_xcoder_for_training(decoder_tasks, side='decoder')

        tw_by_task = decoder.activate(task_id='train_en_is', adapter_ids=None)
        tw_by_component = decoder.activate_by_components(xcoder_ids=['is'], lang='is')

        assert tw_by_task.post_emb_norm is tw_by_component.post_emb_norm
        assert tw_by_task.project_emb is tw_by_component.project_emb

    def test_activate_by_components_caches_wrapper(self):
        """Calling activate_by_components with the same args twice should
        return the same TransformerWrapper (cached, not re-created)."""
        encoder, _ = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        tw1 = encoder.activate_by_components(xcoder_ids=['en'], lang='en')
        tw2 = encoder.activate_by_components(xcoder_ids=['en'], lang='en')
        assert tw1 is tw2

    def test_activate_by_components_unknown_xcoder_id_raises(self):
        """Requesting a component that was never trained should raise KeyError."""
        encoder, _ = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        with pytest.raises(KeyError):
            encoder.activate_by_components(xcoder_ids=['fr'], lang='en')

    def test_activate_by_components_unknown_lang_raises(self):
        """Requesting an embedding for a language that was never trained should raise KeyError."""
        encoder, _ = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        with pytest.raises(KeyError):
            encoder.activate_by_components(xcoder_ids=['en'], lang='fr')

    def test_activate_by_components_multi_layer_stack(self):
        """Test with multiple layer stacks (e.g., xcoder_ids=['de', 'shared'])."""
        tasks = [
            {'corpus_id': 'train_de_is', 'xcoder_ids': ['de', 'shared'], 'lang': 'de'},
            {'corpus_id': 'train_en_is', 'xcoder_ids': ['en', 'shared'], 'lang': 'en'},
        ]
        encoder, parts = _build_stack_xcoder_for_training(tasks, side='encoder')

        # Zero-shot: stack layer0="de" + layer1="shared" with lang="en"
        tw = encoder.activate_by_components(xcoder_ids=['de', 'shared'], lang='en')

        assert tw.attn_layers.attention_layers_stack[0] is parts['attention_layer_blocks'][0]['de']
        assert tw.attn_layers.attention_layers_stack[1] is parts['attention_layer_blocks'][1]['shared']
        assert tw.token_emb is parts['token_embs']['en']


# ===========================================================================
# Test: Forward pass through component-activated TransformerWrapper
# ===========================================================================

class TestComponentActivatedForward:
    """Verify that a TransformerWrapper assembled via activate_by_components
    can actually run a forward pass (no shape mismatches, etc.)."""

    def test_encoder_forward(self):
        encoder, _ = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')
        tw = encoder.activate_by_components(xcoder_ids=['en'], lang='en')

        src = torch.randint(0, VOCAB_SIZE, (2, 10))  # batch=2, seq_len=10
        output = tw(src, return_embeddings=True)
        assert output.shape == (2, 10, DIM)

    def test_decoder_forward(self):
        decoder_tasks = [
            {'corpus_id': 'train_en_is', 'xcoder_ids': ['is'], 'lang': 'is'},
        ]
        decoder, _ = _build_stack_xcoder_for_training(decoder_tasks, side='decoder')
        tw = decoder.activate_by_components(xcoder_ids=['is'], lang='is')

        tgt = torch.randint(0, VOCAB_SIZE, (2, 5))
        logits = tw(tgt)
        # logits shape: (batch, seq_len, vocab_size)
        assert logits.shape == (2, 5, VOCAB_SIZE)

    def test_zero_shot_encoder_forward(self):
        """Forward pass through a zero-shot combination: encoder "de" with lang "en"."""
        encoder, _ = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')
        tw = encoder.activate_by_components(xcoder_ids=['de'], lang='en')

        src = torch.randint(0, VOCAB_SIZE, (2, 10))
        output = tw(src, return_embeddings=True)
        assert output.shape == (2, 10, DIM)

    def test_zero_shot_decoder_forward(self):
        """Forward pass through a zero-shot combination: decoder "en" with lang "is"."""
        decoder_tasks = [
            {'corpus_id': 'train_en_is', 'xcoder_ids': ['is'], 'lang': 'is'},
            {'corpus_id': 'train_is_en', 'xcoder_ids': ['en'], 'lang': 'en'},
        ]
        decoder, _ = _build_stack_xcoder_for_training(decoder_tasks, side='decoder')
        tw = decoder.activate_by_components(xcoder_ids=['en'], lang='is')

        tgt = torch.randint(0, VOCAB_SIZE, (2, 5))
        logits = tw(tgt)
        assert logits.shape == (2, 5, VOCAB_SIZE)


# ===========================================================================
# Test: CLI option validation
# ===========================================================================

class TestTranslateOptsValidation:
    """Tests for the new CLI options: --encoder_id, --decoder_id, --src_lang, --tgt_lang."""

    # Common required args that the parser needs beyond what we're testing
    _REQUIRED_ARGS = [
        '--model', 'model.pt',
        '--src', 'src.txt',
        '--output', 'out.txt',
        '--seed', '42',
        '--max_length', '200',
    ]

    def test_task_id_still_works(self):
        """--task_id alone should still be accepted (backwards compat)."""
        from mammoth.opts import translate_opts
        from mammoth.utils.parse import ArgumentParser

        parser = ArgumentParser()
        parser.translation = True
        translate_opts(parser, dynamic=False)

        parsed = parser.parse_known_args(
            ['--task_id', 'train_en_is'] + self._REQUIRED_ARGS
        )
        assert parsed[0].task_id == 'train_en_is'

    def test_component_flags_accepted(self):
        """--encoder_id, --decoder_id, --src_lang, --tgt_lang should be accepted
        when --task_id is not provided."""
        from mammoth.opts import translate_opts
        from mammoth.utils.parse import ArgumentParser

        parser = ArgumentParser()
        parser.translation = True
        translate_opts(parser, dynamic=False)

        parsed = parser.parse_known_args([
            '--encoder_id', 'de',
            '--decoder_id', 'en',
            '--src_lang', 'de',
            '--tgt_lang', 'en',
        ] + self._REQUIRED_ARGS)
        assert parsed[0].encoder_id == ['de']
        assert parsed[0].decoder_id == ['en']
        assert parsed[0].src_lang == 'de'
        assert parsed[0].tgt_lang == 'en'

    def test_component_flags_multi_layer_stack(self):
        """--encoder_id and --decoder_id should accept multiple values for multi-stack."""
        from mammoth.opts import translate_opts
        from mammoth.utils.parse import ArgumentParser

        parser = ArgumentParser()
        parser.translation = True
        translate_opts(parser, dynamic=False)

        parsed = parser.parse_known_args([
            '--encoder_id', 'de', 'shared',
            '--decoder_id', 'is',
            '--src_lang', 'de',
            '--tgt_lang', 'is',
        ] + self._REQUIRED_ARGS)
        assert parsed[0].encoder_id == ['de', 'shared']


# ===========================================================================
# Test: bin/translate.py zero-shot path
# ===========================================================================

class TestTranslateZeroShotPath:
    """Tests for the zero-shot code path in bin/translate.py.

    These test that when --encoder_id/--decoder_id/--src_lang/--tgt_lang
    are provided (instead of --task_id), the translate function builds
    the correct synthetic TaskSpecs and loads all needed components.
    """

    def test_synthetic_task_has_correct_fields(self):
        """When using component flags, a synthetic TaskSpecs should be created
        with the user-specified encoder_id, decoder_id, src_lang, tgt_lang."""
        from mammoth.distributed import TaskSpecs

        # Simulate what translate() should do in zero-shot mode
        task = TaskSpecs(
            node_rank=0,
            local_rank=0,
            src_lang='de',
            tgt_lang='en',
            encoder_id=['de'],
            decoder_id=['en'],
            corpus_id='__zero_shot__',
            weight=1.0,
            introduce_at_training_step=0,
            corpus_opts={},
            src_vocab=None,
            tgt_vocab=None,
            encoder_adapter_ids=None,
            decoder_adapter_ids=None,
        )
        assert task.src_lang == 'de'
        assert task.tgt_lang == 'en'
        assert task.encoder_id == ['de']
        assert task.decoder_id == ['en']
        assert task.corpus_id == '__zero_shot__'

    def test_zero_shot_task_queue_manager_includes_all_donor_tasks(self):
        """In zero-shot mode, the TQM should include all tasks from the config
        so that all component attention layers and embeddings are available."""
        from mammoth.distributed import TaskSpecs, TaskQueueManager
        from mammoth.distributed.contexts import WorldContext, DeviceContextEnum

        # Two training tasks
        donor_task_en_is = TaskSpecs(
            node_rank=0, local_rank=0,
            src_lang='en', tgt_lang='is',
            encoder_id=['en'], decoder_id=['is'],
            corpus_id='train_en_is', weight=1.0,
            introduce_at_training_step=0, corpus_opts={},
            src_vocab=None, tgt_vocab=None,
            encoder_adapter_ids=None, decoder_adapter_ids=None,
        )
        donor_task_de_is = TaskSpecs(
            node_rank=0, local_rank=0,
            src_lang='de', tgt_lang='is',
            encoder_id=['de'], decoder_id=['is'],
            corpus_id='train_de_is', weight=1.0,
            introduce_at_training_step=0, corpus_opts={},
            src_vocab=None, tgt_vocab=None,
            encoder_adapter_ids=None, decoder_adapter_ids=None,
        )

        world_context = WorldContext(
            context=DeviceContextEnum.CPU,
            n_nodes=1,
            gpus_per_node=0,
        )

        tqm = TaskQueueManager(
            tasks=[donor_task_en_is, donor_task_de_is],
            accum_count=1,
            world_context=world_context,
            task_distribution_strategy_cls=None,
            uses_adapters=False,
        ).global_to_local(node_rank=0, local_rank=0, opts=None)

        my_tasks = tqm.get_my_tasks()
        corpus_ids = {t.corpus_id for t in my_tasks}
        assert 'train_en_is' in corpus_ids
        assert 'train_de_is' in corpus_ids

        # Both "en" and "de" encoder components should be available
        src_langs = set(tqm.get_my_src_langs())
        assert 'en' in src_langs
        assert 'de' in src_langs


# ===========================================================================
# Test: Weights are not re-initialized when creating zero-shot wrappers
# ===========================================================================

class TestNoReinitOnZeroShot:
    """activate_by_components must not re-initialize weights.
    The shared modules should be passed through as-is."""

    def test_embedding_weights_preserved(self):
        """Token embedding weights should be identical (same object) after
        activate_by_components, not freshly initialized."""
        encoder, parts = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        # Record the weight tensor identity
        original_emb_weight = parts['token_embs']['en'].emb.weight

        tw = encoder.activate_by_components(xcoder_ids=['en'], lang='en')
        assert tw.token_emb.emb.weight is original_emb_weight

    def test_attention_layer_weights_preserved(self):
        """Attention layer parameters should be the same objects (not re-initialized)."""
        encoder, parts = _build_stack_xcoder_for_training(ENCODER_TASKS, side='encoder')

        original_attn = parts['attention_layer_blocks'][0]['en']
        original_params = {name: p.data_ptr() for name, p in original_attn.named_parameters()}

        tw = encoder.activate_by_components(xcoder_ids=['en'], lang='en')
        activated_attn = tw.attn_layers.attention_layers_stack[0]

        for name, p in activated_attn.named_parameters():
            assert p.data_ptr() == original_params[name], \
                f"Parameter {name} was re-initialized (different memory address)"
