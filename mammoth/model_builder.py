"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from collections import defaultdict

import mammoth.modules

from mammoth.models.adapters import (
    Adapter,
    EncoderAdapterLayer,
    DecoderAdapterLayer,
)
from mammoth.constants import ModelTask, DefaultTokens
from mammoth.modules.layer_stack_decoder import LayerStackDecoder
from mammoth.modules.layer_stack_encoder import LayerStackEncoder
from mammoth.modules import Embeddings
from mammoth.modules.embeddings import PluggableEmbeddings
from mammoth.modules.util_class import Cast
from mammoth.utils.logging import logger
from mammoth.utils.misc import use_gpu
from mammoth.utils.module_splitter import _combine_ordered_dicts
from mammoth.utils.parse import ArgumentParser

from mammoth.modules.attention_bridge import AttentionBridge


def build_embeddings(opts, vocab, for_encoder=True):
    """
    Args:
        opts: the option in current environment.
        vocab: stoi-ish object.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opts.src_word_vec_size if for_encoder else opts.tgt_word_vec_size
    word_padding_idx = vocab.stoi[DefaultTokens.PAD]
    opts.word_padding_idx = word_padding_idx

    freeze_word_vecs = opts.freeze_word_vecs_enc if for_encoder else opts.freeze_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opts.position_encoding,
        dropout=opts.dropout[0] if type(opts.dropout) is list else opts.dropout,
        word_padding_idx=word_padding_idx,
        word_vocab_size=len(vocab),
        freeze_word_vecs=freeze_word_vecs,
    )
    return emb


def build_encoder(opts, embeddings, task_queue_manager):
    """
    Various encoder dispatcher function.
    Args:
        opts: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    assert opts.encoder_type == 'transformer', 'Only Transformer is supported'
    return LayerStackEncoder.from_opts(opts, embeddings, task_queue_manager)


def build_decoder(opts, embeddings, task_queue_manager):
    """
    Various decoder dispatcher function.
    Args:
        opts: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    assert opts.decoder_type == 'transformer', 'Only Transformer is supported'
    return LayerStackDecoder.from_opts(opts, embeddings, task_queue_manager)


def load_test_multitask_model(opts, model_path=None):
    """If a checkpoint ending with ".pt" returns a full model
    otherwise it builds a bilingual model"""
    if model_path is None:
        model_path = opts.models[0]

    opts.lang_pair = opts.lang_pair if opts.lang_pair else f'{opts.src_lang}-{opts.tgt_lang}'

    if model_path.endswith('.pt'):
        return load_test_model(opts, model_path)
    else:
        checkpoint_modules = [
            (f'encoder.embeddings.embeddings_{opts.src_lang}.', f'src_embeddings_{opts.src_lang}'),
            (f'decoder.embeddings.embeddings_{opts.tgt_lang}.', f'tgt_embeddings_{opts.tgt_lang}'),
            (f'generator.generator_{opts.tgt_lang}.', f'generator_{opts.tgt_lang}'),
            ('attention_bridge.', 'attention_bridge'),
        ]

        for layer_stack_idx, layer_stack_opt in enumerate(opts.stack['encoder']):
            layer_stack_key = layer_stack_opt['id']
            checkpoint_modules.append(
                (
                    f'encoder.encoders.{layer_stack_idx}.{layer_stack_key}.',
                    f'encoder_{layer_stack_idx}_{layer_stack_key}'
                )
            )
            for adapter_group, sub_id in layer_stack_opt.get('adapters', []):
                checkpoint_modules.append(
                    (
                        f'encoder.encoders.{layer_stack_idx}.{layer_stack_key}.adapters.adapter_{adapter_group}_{sub_id}.',    # noqa
                        f'encoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_group}_{sub_id}'
                    )
                )
        for layer_stack_idx, layer_stack_opt in enumerate(opts.stack['decoder']):
            layer_stack_key = layer_stack_opt['id']
            checkpoint_modules.append(
                (
                    f'decoder.decoders.{layer_stack_idx}.{layer_stack_key}.',
                    f'decoder_{layer_stack_idx}_{layer_stack_key}'
                )
            )
            for adapter_group, sub_id in layer_stack_opt.get('adapters', []):
                checkpoint_modules.append(
                    (
                        f'decoder.decoders.{layer_stack_idx}.{layer_stack_key}.adapters.adapter_{adapter_group}_{sub_id}.',    # noqa
                        f'decoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_group}_{sub_id}'
                    )
                )

        model_path = model_path.rstrip('_')
        checkpoint_paths = [
            (prefix, f'{model_path}_{key}.pt') for (prefix, key) in checkpoint_modules
        ]

        opts.model_frame = model_path + '_frame.pt'
        frame = torch.load(opts.model_frame, map_location=lambda storage, loc: storage)

        checkpoint_state_dicts = {
            prefix: torch.load(path, map_location=lambda storage, loc: storage)
            for prefix, path in checkpoint_paths
        }

        combined_state_dict = _combine_ordered_dicts(checkpoint_state_dicts)

        vocabs_dict = {
            'src': frame["vocab"].get(('src', opts.src_lang)),
            'tgt': frame["vocab"].get(('tgt', opts.tgt_lang)),
        }
        # FIXME
        # fields["indices"] = Field(use_vocab=False, dtype=torch.long, sequential=False)

        model_opt = ArgumentParser.ckpt_model_opts(frame['opts'])
        # Avoid functionality on inference
        model_opt.update_vocab = False
        model = create_bilingual_model(
            src_lang=opts.src_lang,
            tgt_lang=opts.tgt_lang,
            opt_stack=opts.stack,
            model_opt=model_opt,
            vocabs_dict=vocabs_dict
            )
        model_params = {name for name, p in model.named_parameters()}
        model_params.update(name for name, p in model.named_buffers())
        for key in set(combined_state_dict.keys()):
            if key not in model_params:
                print(f'Deleting unnecessary key: {key}')
                del combined_state_dict[key]
        for key in model_params:
            if key not in combined_state_dict:
                print(f'Key missing {key}')
        model.load_state_dict(combined_state_dict)
        device = torch.device("cuda" if use_gpu(opts) else "cpu")
        model.to(device)

        model.eval()

        return vocabs_dict, model, model_opt


def load_test_model(opts, model_path=None):
    if model_path is None:
        model_path = opts.models[0]

    if len(opts.models) > 1:
        model_path_enc = opts.models[0]
        checkpoint = torch.load(model_path_enc, map_location=lambda storage, loc: storage)
        model = checkpoint['whole_model']

        model_path_dec = opts.models[1]
        model_dec = torch.load(model_path_dec, map_location=lambda storage, loc: storage)['whole_model']
        model.decoder = model_dec.decoder
        model.generator = model_dec.generator
    else:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = checkpoint['whole_model']

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opts'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocabs = checkpoint['vocab']
    print("VOCABS")
    print(vocabs)
    if opts.gpu != -1:
        device = torch.device("cuda")
        model.to(device)

    lang_pair = opts.lang_pair
    src_lang, tgt_lang = lang_pair.split("-")
    # FIXME
    vocabs_dict = {}
    vocabs_dict['src'] = vocabs[('src', src_lang)]
    vocabs_dict['tgt'] = vocabs[('tgt', tgt_lang)]
    # indices = None  # Field(use_vocab=False, dtype=torch.long, sequential=False)
    # fields["indices"] = indices

    # Avoid functionality on inference
    model_opt.update_vocab = False

    if opts.fp32:
        model.float()
    elif opts.int8:
        if opts.gpu >= 0:
            raise ValueError("Dynamic 8-bit quantization is not supported on GPU")
        torch.quantization.quantize_dynamic(model, inplace=True)
    model.eval()
    model.generator.eval()
    return vocabs_dict, model, model_opt


def create_bilingual_model(
    src_lang, tgt_lang, opt_stack, model_opt, vocabs_dict
):
    """For translation."""
    generators_md = nn.ModuleDict()

    src_emb = build_src_emb(model_opt, vocabs_dict['src'])
    tgt_emb = build_tgt_emb(model_opt, vocabs_dict['tgt'])
    pluggable_src_emb = PluggableEmbeddings({src_lang: src_emb})
    pluggable_tgt_emb = PluggableEmbeddings({tgt_lang: tgt_emb})

    pluggable_src_emb.activate(src_lang)
    pluggable_tgt_emb.activate(tgt_lang)
    encoder = LayerStackEncoder.from_trans_opt(model_opt, pluggable_src_emb, opt_stack)
    decoder = LayerStackDecoder.from_trans_opt(model_opt, pluggable_tgt_emb, opt_stack)
    generator = build_generator(model_opt, len(vocabs_dict['tgt']), tgt_emb)
    generators_md.add_module(f'generator_{tgt_lang}', generator)

    attention_bridge = AttentionBridge.from_opts(model_opt)

    nmt_model = mammoth.models.NMTModel(
        encoder=encoder,
        decoder=decoder,
        attention_bridge=attention_bridge
    )

    if uses_adapters(model_opt):
        logger.info('Creating adapters...')
        create_bilingual_adapters(nmt_model, model_opt, src_lang, tgt_lang, opt_stack)
    else:
        logger.info('Does not use adapters...')
    print('built model:')
    print(nmt_model)

    nmt_model.generator = generators_md
    return nmt_model


def build_src_emb(model_opt, src_vocab):
    # Build embeddings.
    if model_opt.model_type == "text":
        src_emb = build_embeddings(model_opt, src_vocab)
    else:
        src_emb = None
    return src_emb


def build_tgt_emb(model_opt, tgt_vocab):
    # Build embeddings.
    tgt_emb = build_embeddings(model_opt, tgt_vocab, for_encoder=False)

    # if share_embeddings:
    #     tgt_emb.word_lut.weight = src_emb.word_lut.weight

    return tgt_emb


def build_task_specific_model(
    model_opt,
    vocabs_dict,
    device,
    task_queue_manager,
    checkpoint,
):
    logger.info(f'TaskQueueManager: {task_queue_manager}')
    if not model_opt.model_task == ModelTask.SEQ2SEQ:
        raise ValueError(f"Only ModelTask.SEQ2SEQ works - {model_opt.model_task} task")

    src_embs = dict()
    tgt_embs = dict()

    generators_md = nn.ModuleDict()

    # FIXME: it's getting late and I just want this to compile
    for side, lang, _, vocab in task_queue_manager.get_vocabs(side='src', vocabs_dict=vocabs_dict):
        src_emb = build_src_emb(model_opt, vocab)
        src_embs[lang] = src_emb

    pluggable_src_emb = PluggableEmbeddings(src_embs)
    encoder = build_only_enc(model_opt, pluggable_src_emb, task_queue_manager)

    for side, lang, _, vocab in task_queue_manager.get_vocabs(side='tgt', vocabs_dict=vocabs_dict):
        tgt_emb = build_tgt_emb(model_opt, vocab)
        tgt_embs[lang] = tgt_emb
        generator = build_generator(model_opt, len(vocab), tgt_emb)
        generators_md.add_module(f'generator_{lang}', generator)

    pluggable_tgt_emb = PluggableEmbeddings(tgt_embs)
    decoder = build_only_dec(model_opt, pluggable_tgt_emb, task_queue_manager)

    # TODO: implement hierarchical approach to layer sharing
    attention_bridge = AttentionBridge.from_opts(model_opt)

    if model_opt.param_init != 0.0:
        for p in attention_bridge.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in attention_bridge.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        attention_bridge.half()

    nmt_model = mammoth.models.NMTModel(
        encoder=encoder,
        decoder=decoder,
        attention_bridge=attention_bridge
    )
    if uses_adapters(model_opt):
        logger.info('Creating adapters...')
        create_all_adapters(nmt_model, model_opt, task_queue_manager)
    print('built model:')
    print(nmt_model)

    # register a forward hook to keep track of which parameters have valid gradients.
    # p.grad is None can not be used: grad is None only before first update.
    # zero_grad typically sets the grad to zero, not to None.
    # While zero_grad takes a flag set_to_none, it is not reliably forwarded by various optimizers.
    def has_grad_hook(module, input, output) -> None:
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                # NB: we're looking at whether gradient will/has been computed, which is only the
                # case when the module is training.
                param.has_grad = module.training

    for module in nmt_model.modules():
        module.register_forward_hook(has_grad_hook)
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                param.has_grad = False
    for module in generators_md.modules():
        module.register_forward_hook(has_grad_hook)
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                param.has_grad = False

    return nmt_model, generators_md


def build_only_enc(model_opt, src_emb, task_queue_manager):
    """Truly only builds encoder: no embeddings"""
    encoder = build_encoder(model_opt, src_emb, task_queue_manager)
    if model_opt.param_init != 0.0:
        for p in encoder.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        encoder.half()

    return encoder


def build_only_dec(model_opt, tgt_emb, task_queue_manager):
    decoder = build_decoder(model_opt, tgt_emb, task_queue_manager)

    if model_opt.param_init != 0.0:
        for p in decoder.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        decoder.half()

    return decoder


def build_generator(model_opt, n_tgts, tgt_emb):
    # Build Generator.
    assert not model_opt.copy_attn, 'copy_attn not supported'
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, n_tgts), Cast(torch.float32), gen_func
    )

    if model_opt.share_decoder_embeddings:
        generator[0].weight = tgt_emb.word_lut.weight

    if model_opt.param_init != 0.0:
        for p in generator.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    return generator


# TODO: confirm this was dead code
# def use_embeddings_from_checkpoint(fields, model, generator, checkpoint):
#     # Update vocabulary embeddings with checkpoint embeddings
#     logger.info("Updating vocabulary embeddings with checkpoint embeddings")
#     # Embedding layers
#     enc_emb_name = "encoder.embeddings.make_embedding.emb_luts.0.weight"
#     dec_emb_name = "decoder.embeddings.make_embedding.emb_luts.0.weight"
#
#     for field_name, emb_name in [("src", enc_emb_name), ("tgt", dec_emb_name)]:
#         if emb_name not in checkpoint["model"]:
#             continue
#         multifield = fields[field_name]
#         checkpoint_multifield = checkpoint["vocab"][field_name]
#         for (name, field), (checkpoint_name, checkpoint_field) in zip(multifield, checkpoint_multifield):
#             new_tokens = []
#             for i, tok in enumerate(field.vocab.itos):
#                 if tok in checkpoint_field.vocab.stoi:
#                     old_i = checkpoint_field.vocab.stoi[tok]
#                     model.state_dict()[emb_name][i] = checkpoint["model"][emb_name][old_i]
#                     if field_name == "tgt":
#                         generator.state_dict()["0.weight"][i] = checkpoint["generator"]["0.weight"][old_i]
#                         generator.state_dict()["0.bias"][i] = checkpoint["generator"]["0.bias"][old_i]
#                 else:
#                     # Just for debugging purposes
#                     new_tokens.append(tok)
#             logger.info("%s: %d new tokens" % (name, len(new_tokens)))
#         # Remove old vocabulary associated embeddings
#         del checkpoint["model"][emb_name]
#     del checkpoint["generator"]["0.weight"], checkpoint["generator"]["0.bias"]


def build_base_model_langspec(
    model_opt,
    vocabs_dict,
    gpu,
    task_queue_manager,
    checkpoint=None,
):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`mammoth.utils.parse.ArgumentParser`.
        vocabs_dict (dict[str, mammoth.inputters.Vocab]):
            `Vocab` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build Model
    logger.info("MODEL BUILDER")
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(device)
    model, generators_md = build_task_specific_model(
        model_opt=model_opt,
        vocabs_dict=vocabs_dict,
        device=device,
        task_queue_manager=task_queue_manager,
        checkpoint=checkpoint,
    )

    model.generator = generators_md
    model.to(device)

    return model, generators_md


def uses_adapters(opts):
    return 'adapters' in opts and opts.adapters


def create_all_adapters(model, opts, task_queue_manager):
    my_enc_adapter_ids = set()
    my_dec_adapter_ids = set()
    adapter_to_encoder_ids = defaultdict(set)
    adapter_to_decoder_ids = defaultdict(set)
    for task in task_queue_manager.get_tasks():
        for adapter_id in task.encoder_adapter_ids:
            adapter_id = tuple(adapter_id)
            my_enc_adapter_ids.add(adapter_id)
            adapter_to_encoder_ids[adapter_id].add(tuple(task.encoder_id))
        for adapter_id in task.decoder_adapter_ids:
            adapter_id = tuple(adapter_id)
            my_dec_adapter_ids.add(adapter_id)
            adapter_to_decoder_ids[adapter_id].add(tuple(task.decoder_id))
    _create_adapters(
        model,
        opts,
        my_enc_adapter_ids,
        adapter_to_encoder_ids,
        my_dec_adapter_ids,
        adapter_to_decoder_ids,
    )


def create_bilingual_adapters(model, opts, src_lang, tgt_lang, opt_stack):
    my_enc_adapter_ids = []
    my_dec_adapter_ids = []
    adapter_to_encoder_ids = {}
    adapter_to_decoder_ids = {}

    enc_ids = [layer_stack_opts['id'] for layer_stack_opts in opt_stack['encoder']]
    dec_ids = [layer_stack_opts['id'] for layer_stack_opts in opt_stack['decoder']]

    for layer_stack_index, layer_stack_opts in enumerate(opt_stack['encoder']):
        for group_id, sub_id in layer_stack_opts.get('adapters', []):
            adapter_id = (layer_stack_index, group_id, sub_id)
            my_enc_adapter_ids.append(adapter_id)
            adapter_to_encoder_ids[adapter_id] = [enc_ids]

    for layer_stack_index, layer_stack_opts in enumerate(opt_stack['decoder']):
        for group_id, sub_id in layer_stack_opts.get('adapters', []):
            adapter_id = (layer_stack_index, group_id, sub_id)
            my_dec_adapter_ids.append(adapter_id)
            adapter_to_decoder_ids[adapter_id] = [dec_ids]

    _create_adapters(
        model,
        opts,
        my_enc_adapter_ids,
        adapter_to_encoder_ids,
        my_dec_adapter_ids,
        adapter_to_decoder_ids,
    )


def _create_adapters(
    model,
    opts,
    my_enc_adapter_ids,
    adapter_to_encoder_ids,
    my_dec_adapter_ids,
    adapter_to_decoder_ids,
):
    my_enc_adapter_ids = [tuple(item) for item in my_enc_adapter_ids]
    my_dec_adapter_ids = [tuple(item) for item in my_dec_adapter_ids]
    for adapter_group, adapter_opts in opts.adapters['encoder'].items():
        layer_stack_index = adapter_opts['layer_stack_index']
        for sub_id in adapter_opts['ids']:
            adapter_id_long = (layer_stack_index, adapter_group, sub_id)
            if adapter_id_long not in my_enc_adapter_ids:
                continue
            adapter = Adapter(adapter_group, sub_id)
            input_dim = opts.rnn_size
            hidden_dim = adapter_opts['hidden_size']

            # all stacks to which this adapter should be added
            adapted_stacks = set(
                stacks[layer_stack_index] for stacks in adapter_to_encoder_ids[adapter_id_long]
            )
            adapter_cls = EncoderAdapterLayer

            for layer_idx in adapter_opts['layers']:
                adapter.add_layer(
                    layer_idx,
                    adapter_cls(input_dim, hidden_dim, pfeiffer=False, init='small')
                )
            model.encoder.add_adapter(
                adapter_group=adapter_group,
                sub_id=sub_id,
                adapter=adapter,
                layer_stack_index=layer_stack_index,
                module_ids=adapted_stacks,
            )
    for adapter_group, adapter_opts in opts.adapters['decoder'].items():
        layer_stack_index = adapter_opts['layer_stack_index']
        for sub_id in adapter_opts['ids']:
            adapter_id_long = (layer_stack_index, adapter_group, sub_id)
            if adapter_id_long not in my_dec_adapter_ids:
                continue
            adapter = Adapter(adapter_group, sub_id)
            input_dim = opts.rnn_size
            hidden_dim = adapter_opts['hidden_size']

            adapted_stacks = set(
                stacks[layer_stack_index] for stacks in adapter_to_decoder_ids[adapter_id_long]
            )
            adapter_cls = DecoderAdapterLayer

            for layer_idx in adapter_opts['layers']:
                adapter.add_layer(
                    layer_idx,
                    adapter_cls(input_dim, hidden_dim, pfeiffer=False, init='small')
                )
            model.decoder.add_adapter(
                adapter_group=adapter_group,
                sub_id=sub_id,
                adapter=adapter,
                layer_stack_index=layer_stack_index,
                module_ids=adapted_stacks,
            )


def build_model(model_opt, opts, vocabs_dict, task_queue_manager, checkpoint):
    logger.info('Building model...')
    model, generators_md = build_base_model_langspec(
        model_opt=model_opt,
        vocabs_dict=vocabs_dict,
        gpu=use_gpu(opts),
        task_queue_manager=task_queue_manager,
        checkpoint=checkpoint,
    )
    # logger.info(model)
    logger.info('Building model - done!')
    return model, generators_md
