"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from collections import defaultdict
from torchtext.legacy.data import Field

import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings
from onmt.modules.embeddings import PluggableEmbeddings
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from onmt.utils.module_splitter import create_bilingual_statedict
from onmt.constants import ModelTask

from onmt.attention_bridge import AttentionBridge


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]
    opt.word_padding_idx = word_padding_idx

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    freeze_word_vecs = opt.freeze_word_vecs_enc if for_encoder else opt.freeze_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        freeze_word_vecs=freeze_word_vecs,
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_multitask_model(opt, model_path=None):
    """If a checkpoint ending with ".pt" returns a full model
    otherwise it builds a bilingual model"""
    if model_path is None:
        model_path = opt.models[0]

    opt.lang_pair = opt.lang_pair if opt.lang_pair else f'{opt.src_lang}-{opt.tgt_lang}'

    if model_path.endswith('.pt'):
        return load_test_model(opt, model_path)
    else:
        model_path = model_path if model_path.endswith('_') else f'{model_path}_'
        enc_path = model_path + opt.src_lang + '_enc.pt'
        dec_path = model_path + opt.tgt_lang + '_dec.pt'
        opt.generator = model_path + opt.tgt_lang + '_gen.pt' if opt.generator is None else opt.generator
        opt.bridge = model_path + 'bridge.pt' if opt.bridge is None else opt.bridge
        opt.model_frame = model_path + 'frame.pt' if opt.model_frame is None else opt.model_frame

        encoder = torch.load(enc_path, map_location=lambda storage, loc: storage)
        decoder = torch.load(dec_path, map_location=lambda storage, loc: storage)
        bridge = torch.load(opt.bridge, map_location=lambda storage, loc: storage)
        generator = torch.load(opt.generator, map_location=lambda storage, loc: storage)
        frame = torch.load(opt.model_frame, map_location=lambda storage, loc: storage)

        ckpt_state_dict = create_bilingual_statedict(
            src_lang=opt.src_lang,
            tgt_lang=opt.tgt_lang,
            enc_module=encoder,
            dec_module=decoder,
            ab_module=bridge,
            gen_module=generator,
        )

        fields = {
            'src': frame["vocab"].get(('src', opt.src_lang))['src'],
            'tgt': frame["vocab"].get(('tgt', opt.tgt_lang))['tgt'],
        }
        fields["indices"] = Field(use_vocab=False, dtype=torch.long, sequential=False)

        model_opt = ArgumentParser.ckpt_model_opts(frame['opt'])
        # Avoid functionality on inference
        model_opt.update_vocab = False
        model = create_bilingual_model(src_lang=opt.src_lang, tgt_lang=opt.tgt_lang, model_opt=model_opt, fields=fields)
        model.load_state_dict(ckpt_state_dict)
        device = torch.device("cuda" if use_gpu(opt) else "cpu")
        model.to(device)

        model.eval()

        return fields, model, model_opt


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]

    if len(opt.models) > 1:
        model_path_enc = opt.models[0]
        checkpoint = torch.load(model_path_enc, map_location=lambda storage, loc: storage)
        model = checkpoint['whole_model']

        model_path_dec = opt.models[1]
        model_dec = torch.load(model_path_dec, map_location=lambda storage, loc: storage)['whole_model']
        model.decoder = model_dec.decoder
        model.generator = model_dec.generator
    else:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = checkpoint['whole_model']

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    fields_dict = checkpoint['vocab']
    print("FIELDS")
    print(fields_dict)
    if opt.gpu != -1:
        device = torch.device("cuda")
        model.to(device)

    lang_pair = opt.lang_pair
    src_lang, tgt_lang = lang_pair.split("-")
    fields = {}
    fields['src'] = fields_dict[('src', src_lang)]['src']
    fields['tgt'] = fields_dict[('tgt', tgt_lang)]['tgt']
    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = indices

    # Avoid functionality on inference
    model_opt.update_vocab = False

    if opt.fp32:
        model.float()
    elif opt.int8:
        if opt.gpu >= 0:
            raise ValueError("Dynamic 8-bit quantization is not supported on GPU")
        torch.quantization.quantize_dynamic(model, inplace=True)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def create_bilingual_model(src_lang, tgt_lang, model_opt, fields):
    """For translation - state dict to be loaded to this model."""

    encoder = nn.ModuleDict()
    decoder = nn.ModuleDict()
    generator = nn.ModuleDict()

    src_emb = build_src_emb(model_opt, fields)
    tgt_emb = build_tgt_emb(model_opt, fields)
    pluggable_src_emb = PluggableEmbeddings({f'{src_lang}': src_emb})
    pluggable_tgt_emb = PluggableEmbeddings({f'{tgt_lang}': tgt_emb})

    pluggable_src_emb.activate(src_lang)
    pluggable_tgt_emb.activate(tgt_lang)
    encoder.add_module(f'encoder{src_lang}', build_only_enc(model_opt, pluggable_src_emb))
    decoder.add_module(f'decoder{tgt_lang}', build_only_dec(model_opt, pluggable_tgt_emb))
    generator.add_module(f'generator{tgt_lang}', build_generator(model_opt, fields, tgt_emb))

    attention_bridge = AttentionBridge.from_opt(model_opt)

    nmt_model = onmt.models.NMTModel(encoder=encoder, decoder=decoder, attention_bridge=attention_bridge)

    nmt_model.generator = generator
    return nmt_model


def build_src_emb(model_opt, fields):
    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None
    return src_emb


def build_tgt_emb(model_opt, fields):
    # Build embeddings.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    # if share_embeddings:
    #     tgt_emb.word_lut.weight = src_emb.word_lut.weight

    return tgt_emb


def build_task_specific_model(
    model_opt,
    fields_dict,
    device,
    scheduler,
    checkpoint,
):
    logger.info(f'Scheduler: {scheduler}')
    if not model_opt.model_task == ModelTask.SEQ2SEQ:
        raise ValueError(f"Only ModelTask.SEQ2SEQ works - {model_opt.model_task} task")

    src_embs_by_encoder = defaultdict(dict)
    tgt_embs_by_decoder = defaultdict(dict)

    encoders_md = nn.ModuleDict()
    decoders_md = nn.ModuleDict()
    generators_md = nn.ModuleDict()

    for side, lang, encoder_id, fields in scheduler.get_fields(side='src', fields_dict=fields_dict):
        src_emb = build_src_emb(model_opt, fields)
        src_embs_by_encoder[encoder_id][lang] = src_emb

    for encoder_id in scheduler.get_encoders():
        pluggable_src_emb = PluggableEmbeddings(src_embs_by_encoder[encoder_id])
        encoder = build_only_enc(model_opt, pluggable_src_emb)
        encoders_md.add_module(f'encoder{encoder_id}', encoder)

    for side, lang, decoder_id, fields in scheduler.get_fields(side='tgt', fields_dict=fields_dict):
        tgt_emb = build_tgt_emb(model_opt, fields)
        tgt_embs_by_decoder[decoder_id][lang] = tgt_emb
        generator = build_generator(model_opt, fields, tgt_emb)
        generators_md.add_module(f'generator{lang}', generator)

    for decoder_id in scheduler.get_decoders():
        pluggable_tgt_emb = PluggableEmbeddings(tgt_embs_by_decoder[decoder_id])
        decoder = build_only_dec(model_opt, pluggable_tgt_emb)
        decoders_md.add_module(f'decoder{decoder_id}', decoder)

    # TODO: implement hierarchical approach to layer sharing
    attention_bridge = AttentionBridge.from_opt(model_opt)

    if model_opt.param_init != 0.0:
        for p in attention_bridge.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in attention_bridge.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        attention_bridge.half()

    nmt_model = onmt.models.NMTModel(encoder=encoders_md, decoder=decoders_md, attention_bridge=attention_bridge)
    return nmt_model, generators_md


def build_only_enc(model_opt, src_emb):
    """Truly only builds encoder: no embeddings"""
    encoder = build_encoder(model_opt, src_emb)
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


def build_only_dec(model_opt, tgt_emb):
    decoder = build_decoder(model_opt, tgt_emb)

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


def build_generator(model_opt, fields, tgt_emb):
    # Build Generator.
    assert not model_opt.copy_attn, 'copy_attn not supported'
    if model_opt.generator_function == "sparsemax":
        gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
    else:
        gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].base_field.vocab)), Cast(torch.float32), gen_func
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


def use_embeddings_from_checkpoint(fields, model, generator, checkpoint):
    # Update vocabulary embeddings with checkpoint embeddings
    logger.info("Updating vocabulary embeddings with checkpoint embeddings")
    # Embedding layers
    enc_emb_name = "encoder.embeddings.make_embedding.emb_luts.0.weight"
    dec_emb_name = "decoder.embeddings.make_embedding.emb_luts.0.weight"

    for field_name, emb_name in [("src", enc_emb_name), ("tgt", dec_emb_name)]:
        if emb_name not in checkpoint["model"]:
            continue
        multifield = fields[field_name]
        checkpoint_multifield = checkpoint["vocab"][field_name]
        for (name, field), (checkpoint_name, checkpoint_field) in zip(multifield, checkpoint_multifield):
            new_tokens = []
            for i, tok in enumerate(field.vocab.itos):
                if tok in checkpoint_field.vocab.stoi:
                    old_i = checkpoint_field.vocab.stoi[tok]
                    model.state_dict()[emb_name][i] = checkpoint["model"][emb_name][old_i]
                    if field_name == "tgt":
                        generator.state_dict()["0.weight"][i] = checkpoint["generator"]["0.weight"][old_i]
                        generator.state_dict()["0.bias"][i] = checkpoint["generator"]["0.bias"][old_i]
                else:
                    # Just for debugging purposes
                    new_tokens.append(tok)
            logger.info("%s: %d new tokens" % (name, len(new_tokens)))
        # Remove old vocabulary associated embeddings
        del checkpoint["model"][emb_name]
    del checkpoint["generator"]["0.weight"], checkpoint["generator"]["0.bias"]


def build_base_model_langspec(
    model_opt,
    fields_dict,
    gpu,
    scheduler,
    checkpoint=None,
):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
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
        fields_dict=fields_dict,
        device=device,
        scheduler=scheduler,
        checkpoint=checkpoint,
    )

    model.generator = generators_md
    model.to(device)

    return model, generators_md


def build_model(model_opt, opt, fields_dict, scheduler, checkpoint):
    logger.info('Building model...')
    model, generators_md = build_base_model_langspec(
        model_opt=model_opt,
        fields_dict=fields_dict,
        gpu=use_gpu(opt),
        scheduler=scheduler,
        checkpoint=checkpoint,
    )
    # logger.info(model)
    logger.info('Building model - done!')
    return model, generators_md
