"""
Parity oracle: verify the model under test produces outputs that match the
golden fixtures captured from the original x-transformers model.

Run golden/generate.py once to create the fixtures, then run this suite
after every implementation phase to confirm numerical parity.
"""
import os
import pytest
import torch
import mammoth
import mammoth.opts
from mammoth.model_builder import build_model
from mammoth.utils.parse import ArgumentParser
from mammoth.distributed.tasks import (
    TaskSpecs,
    TaskQueueManager,
    RoundRobinTaskDistributionStrategy,
)
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum
from mammoth.utils.loss import build_loss_function

from mammoth.tests.golden.config import (
    SEED,
    MODEL_DIM,
    ENC_LAYERS,
    DEC_LAYERS,
    X_TRANSFORMERS_OPTS,
    VOCABS,
    make_inputs,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'golden')

_parser = ArgumentParser(description='golden')
mammoth.opts.model_opts(_parser)
mammoth.opts._add_train_general_opts(_parser)

_BASE_ARGS = '-tasks dummy -node_rank 0'


def _fixtures_exist():
    return all(
        os.path.exists(os.path.join(FIXTURE_DIR, f))
        for f in ('encoder_output.pt', 'decoder_out.pt', 'loss.pt')
    )


def _make_opts():
    enc_arg = ' '.join(str(d) for d in ENC_LAYERS)
    dec_arg = ' '.join(str(d) for d in DEC_LAYERS)
    raw = _parser.parse_known_args(
        f'{_BASE_ARGS} -model_dim {MODEL_DIM} -enc_layers {enc_arg} -dec_layers {dec_arg}'.split(),
        strict=False,
    )[0]
    raw.x_transformers_opts = dict(X_TRANSFORMERS_OPTS)
    ArgumentParser.validate_model_opts(raw)
    return raw


def _build_tqm_and_vocabs(opts):
    task = TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='a',
        tgt_lang='b',
        encoder_id=['enc'],
        decoder_id=['dec'],
        corpus_id='a-b',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts={},
        src_vocab=VOCABS[('src', 'a')],
        tgt_vocab=VOCABS[('tgt', 'b')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )
    world_context = WorldContext(DeviceContextEnum.MULTI_GPU, n_nodes=1, gpus_per_node=2)
    tqm = TaskQueueManager(
        tasks=[task],
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=RoundRobinTaskDistributionStrategy,
        uses_adapters=False,
    ).global_to_local(node_rank=0, local_rank=0, opts=opts)

    vocabs_dict = {
        (side, lang): vocab
        for (side, lang, _, vocab) in tqm.get_my_vocabs('src', VOCABS)
    }
    vocabs_dict.update({
        (side, lang): vocab
        for (side, lang, _, vocab) in tqm.get_my_vocabs('tgt', VOCABS)
    })
    tqm.create_all_distributed_components(use_attention_bridge=False)
    return tqm, vocabs_dict, task


@pytest.fixture(scope='module')
def model_and_data():
    """Build model once for the whole module."""
    if not _fixtures_exist():
        pytest.skip('Golden fixtures not found — run mammoth/tests/golden/generate.py first')

    torch.manual_seed(SEED)
    opts = _make_opts()
    tqm, vocabs_dict, task = _build_tqm_and_vocabs(opts)
    model = build_model(opts, opts, vocabs_dict, task_queue_manager=tqm)
    model.eval()

    src, decoder_input, src_mask, tgt_labels = make_inputs()
    return model, task, vocabs_dict, src, decoder_input, src_mask, tgt_labels


def test_encoder_output(model_and_data):
    model, task, vocabs_dict, src, decoder_input, src_mask, tgt_labels = model_and_data
    golden = torch.load(os.path.join(FIXTURE_DIR, 'encoder_output.pt'), weights_only=True)

    with torch.no_grad():
        active_enc = model.encoder.activate(
            task_id=task.corpus_id,
            adapter_ids=task.encoder_adapter_ids,
        )
        encoder_output = active_enc(src, mask=src_mask, return_embeddings=True)

    assert encoder_output.shape == golden.shape, (
        f'Shape mismatch: got {encoder_output.shape}, expected {golden.shape}'
    )
    assert torch.allclose(encoder_output, golden, atol=1e-5), (
        f'Encoder output differs from golden (max delta '
        f'{(encoder_output - golden).abs().max().item():.2e})'
    )


def test_decoder_output(model_and_data):
    model, task, vocabs_dict, src, decoder_input, src_mask, tgt_labels = model_and_data
    golden = torch.load(os.path.join(FIXTURE_DIR, 'decoder_out.pt'), weights_only=True)
    metadata = task.get_serializable_metadata()

    with torch.no_grad():
        logits, decoder_output = model(src, decoder_input, src_mask, metadata=metadata)

    for name, got, exp in [
        ('logits', logits, golden['logits']),
        ('decoder_output', decoder_output, golden['decoder_output']),
    ]:
        assert got.shape == exp.shape, f'{name} shape mismatch: {got.shape} vs {exp.shape}'
        assert torch.allclose(got, exp, atol=1e-5), (
            f'{name} differs from golden '
            f'(max delta {(got - exp).abs().max().item():.2e})'
        )


def test_loss(model_and_data):
    model, task, vocabs_dict, src, decoder_input, src_mask, tgt_labels = model_and_data
    golden_loss = torch.load(os.path.join(FIXTURE_DIR, 'loss.pt'), weights_only=True)
    metadata = task.get_serializable_metadata()

    with torch.no_grad():
        logits, _ = model(src, decoder_input, src_mask, metadata=metadata)
        tgt_vocab = vocabs_dict[('tgt', 'b')]
        loss_fn = build_loss_function(tgt_vocab, label_smoothing=0.0)
        vocab_size = logits.size(-1)
        loss = loss_fn(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))

    assert torch.allclose(loss, golden_loss, atol=1e-4), (
        f'Loss differs: got {loss.item():.6f}, expected {golden_loss.item():.6f}'
    )
