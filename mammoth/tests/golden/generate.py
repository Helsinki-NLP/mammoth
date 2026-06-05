"""
Capture golden fixtures from the current (x-transformers) model.
Run once before starting the native rewrite; commit the resulting .pt files.

Usage (from the repo root):
    python -m mammoth.tests.golden.generate
"""
import os
import torch
import mammoth
import mammoth.opts
from mammoth.model_builder import build_model, build_xcoder
from mammoth.utils.parse import ArgumentParser
from mammoth.distributed.components import Side
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

FIXTURE_DIR = os.path.dirname(__file__)

_parser = ArgumentParser(description='golden')
mammoth.opts.model_opts(_parser)
mammoth.opts._add_train_general_opts(_parser)

_BASE_ARGS = '-tasks dummy -node_rank 0'


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


def generate():
    torch.manual_seed(SEED)
    opts = _make_opts()

    tqm, vocabs_dict, task = _build_tqm_and_vocabs(opts)
    metadata = task.get_serializable_metadata()

    model = build_model(opts, opts, vocabs_dict, task_queue_manager=tqm)
    model.eval()

    src, decoder_input, src_mask, tgt_labels = make_inputs()

    with torch.no_grad():
        # 1. Encoder forward
        active_enc = model.encoder.activate(
            task_id=task.corpus_id,
            adapter_ids=task.encoder_adapter_ids,
        )
        encoder_output = active_enc(src, mask=src_mask, return_embeddings=True)

        # 2. Full model forward (logits + decoder embeddings)
        logits, decoder_output = model(src, decoder_input, src_mask, metadata=metadata)

        # 3. Single train-step loss
        tgt_vocab = vocabs_dict[('tgt', 'b')]
        loss_fn = build_loss_function(tgt_vocab, label_smoothing=0.0)
        vocab_size = logits.size(-1)
        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            tgt_labels.reshape(-1),
        )

    torch.save(encoder_output, os.path.join(FIXTURE_DIR, 'encoder_output.pt'))
    torch.save({'logits': logits, 'decoder_output': decoder_output},
               os.path.join(FIXTURE_DIR, 'decoder_out.pt'))
    torch.save(loss, os.path.join(FIXTURE_DIR, 'loss.pt'))

    print(f'encoder_output shape : {encoder_output.shape}')
    print(f'logits shape         : {logits.shape}')
    print(f'decoder_output shape : {decoder_output.shape}')
    print(f'loss                 : {loss.item():.6f}')
    print(f'Fixtures written to  : {FIXTURE_DIR}')


if __name__ == '__main__':
    generate()
