"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser
from onmt.constants import ModelTask

from onmt.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls
from onmt.inputters.fields import build_dynamic_fields, save_fields, \
    load_fields
from collections import OrderedDict
from onmt.attention_bridge import AttentionBridge
from torchtext.legacy.data import Field

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

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    freeze_word_vecs = opt.freeze_word_vecs_enc if for_encoder \
        else opt.freeze_word_vecs_dec

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
        freeze_word_vecs=freeze_word_vecs
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
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]

    if len(opt.models) > 1:
        model_path_enc = opt.models[0]
        checkpoint = torch.load(model_path_enc, map_location=lambda storage, loc: storage) 
        model = checkpoint['whole_model']
        print("PRIMA 1")
        for name, param in model.decoder["decodercs"].named_parameters():
            print(str(name)+" "+str(param[0:10]))

        model_path_dec = opt.models[1]
        model_dec = torch.load(model_path_dec, map_location=lambda storage, loc: storage)['whole_model']
        print("DOPO 1")
        for name, param in model_dec.decoder["decodercs"].named_parameters():
            print(str(name)+" "+str(param[0:10]))
        print("DOPO 2")
        model.decoder = model_dec.decoder
        model.generator = model_dec.generator
    else:
        checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
        model = checkpoint['whole_model']


    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    #fields = checkpoint['vocab']
    Fields_dict = checkpoint['vocab']
    print("FIELDS")
    print(Fields_dict)
#    model = checkpoint['whole_model']
    device = torch.device("cuda")
    model.to(device)

    langpair = opt.lang_pair
    langENC = str(langpair).split("-")[0]
    langDEC = str(langpair).split("-")[1]
    fields = {}
    if langpair in Fields_dict:
        fields = Fields_dict[langpair]
    else:
        #we can omit it, but ok
        encDone = False
        decDone = False
        for langpairFields in Fields_dict:
            if encDone and decDone:
                break 
            LANGsrc_tgt = langpairFields.split("-")
            if LANGsrc_tgt[0] == langENC and not encDone:
                fields["src"] = Fields_dict[langpairFields]["src"]
                encDone = True
            if LANGsrc_tgt[1] == langDEC and not decDone:
                fields["tgt"] = Fields_dict[langpairFields]["tgt"]
                decDone = True
        indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
        fields["indices"] = indices

    # Avoid functionality on inference
    model_opt.update_vocab = False

    print("====")
    print(fields)
#    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
#                             opt.gpu)
    if opt.fp32:
        model.float()
    elif opt.int8:
        if opt.gpu >= 0:
            raise ValueError(
                "Dynamic 8-bit quantization is not supported on GPU")
        torch.quantization.quantize_dynamic(model, inplace=True)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt



def build_src_emb(model_opt, fields):
    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None
    return src_emb


def build_encoder_with_embeddings(model_opt, fields):
    # Build encoder.
    src_emb = build_src_emb(model_opt, fields)
    encoder = build_encoder(model_opt, src_emb)
    return encoder, src_emb


def build_decoder_with_embeddings(
    model_opt, fields, share_embeddings=False, src_emb=None
):
    # Build embeddings.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    if share_embeddings:
        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    # Build decoder.
    decoder = build_decoder(model_opt, tgt_emb)
    return decoder, tgt_emb




def build_task_specific_model(model_opt, Fields_dict, device, checkpoint, node_rank, device_rank):
#def build_task_specific_model(model_opt, fields, device, checkpoint, node_rank, device_rank):
    # Share the embedding matrix - preprocess with share_vocab required.
#    if model_opt.share_embeddings:
#        # src/tgt vocab should be the same if `-share_vocab` is specified.
#        assert (
#            fields["src"].base_field.vocab == fields["tgt"].base_field.vocab
#        ), "preprocess with -share_vocab if you use share_embeddings"

#    if model_opt.model_task == ModelTask.SEQ2SEQ:
#        encoder, src_emb = build_encoder_with_embeddings(model_opt, fields)
#        decoder, _ = build_decoder_with_embeddings(
#            model_opt,
#            fields,
#            share_embeddings=model_opt.share_embeddings,
#            src_emb=src_emb,
#        )
#        return onmt.models.NMTModel(encoder=encoder, decoder=decoder)
    #deviceCPU = torch.device("cpu")
    device = torch.device("cuda")
    #device0 = torch.device("cuda", 0)
    #device1 = torch.device("cuda", 1)
    #device2 = torch.device("cuda", 2)
    #device3 = torch.device("cuda", 3)
    logger.info("NODE RANK DEVICE RANK")
    logger.info(str(node_rank) +" "+str(device_rank))
    #Fields_dict = None
    if model_opt.model_task == ModelTask.SEQ2SEQ:
        encoders_md = None
        decoders_md = None
#        Fields_dict = OrderedDict()
        #model_opt.node_gpu_langs
        #node_gpu_assignment.txt
        #0 0 de-en en-fi fi-fr
#        if model_opt.target_langs:
        if model_opt.node_gpu_langs:

            logger.info("TARGET LANGS BUILDER") 

            encoders_md = nn.ModuleDict()
            decoders_md = nn.ModuleDict()
            generators_md = nn.ModuleDict()

            sourceLangs = set()
            targetLangs = set()
            node_gpu_langsFile = open(model_opt.node_gpu_langs, 'rt')
            for line in node_gpu_langsFile:
                lang = line.strip()
                node_gpu_langs = lang.split(" ")
                nodeidx = int(node_gpu_langs[0])
                gpuidx = int(node_gpu_langs[1])
                if nodeidx == node_rank and gpuidx==device_rank:
                    for src_tgt_lang in node_gpu_langs[2:]:
                        LANGsrc_tgt = src_tgt_lang.split("-")
                        logger.info(LANGsrc_tgt)
                        #fields, transforms_cls = prepare_fields_transforms(opt, LANGsrc_tgt[0], LANGsrc_tgt[1])
                        #Fields_dict[src_tgt_lang] = fields
                        fields = Fields_dict[src_tgt_lang]
                        if not LANGsrc_tgt[0] in sourceLangs:
                            sourceLangs.add(LANGsrc_tgt[0])
                            encoder, src_emb = buildOnlyEnc(model_opt, fields)
                            encoders_md.add_module('encoder{0}'.format(LANGsrc_tgt[0]), encoder)
                        if not LANGsrc_tgt[1] in targetLangs:
                            targetLangs.add(LANGsrc_tgt[1])
                            decoder, generator = buildDecGen(model_opt, fields, None)
                            decoders_md.add_module('decoder{0}'.format(LANGsrc_tgt[1]), decoder)
                            generators_md.add_module('generator{0}'.format(LANGsrc_tgt[1]), generator)


            node_gpu_langsFile.close()

                        
        #model = onmt.models.NMTModel(encoder=encoders_md, decoder=decoders_md)
        #model.to(device)
        attention_bridge = AttentionBridge(model_opt.rnn_size, model_opt.attention_heads, model_opt)
        
        if model_opt.param_init != 0.0:
            for p in attention_bridge.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in attention_bridge.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
        if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
            attention_bridge.half()
        
        return onmt.models.NMTModel(encoder=encoders_md, decoder=decoders_md, attention_bridge=attention_bridge), generators_md
    else:
        raise ValueError(f"Only ModelTask.SEQ2SEQ works - {model_opt.model_task} task")

def buildOnlyEnc(model_opt, fields):
    encoder, src_emb = build_encoder_with_embeddings(model_opt, fields)
    if model_opt.param_init != 0.0:
        for p in encoder.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in encoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        encoder.half()
        
    return encoder, src_emb

def buildDecGen(model_opt, fields, src_emb):
    decoder, _ = build_decoder_with_embeddings(
        model_opt,
        fields,
        share_embeddings=model_opt.share_embeddings,
        src_emb=src_emb,
        )

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(nn.Linear(model_opt.dec_rnn_size,len(fields["tgt"].base_field.vocab)),Cast(torch.float32),gen_func)
        if model_opt.share_decoder_embeddings:
            generator[0].weight = model.decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator.linear.weight = model.decoder.embeddings.word_lut.weight

    if model_opt.param_init != 0.0:
        for p in decoder.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        for p in generator.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    if model_opt.param_init_glorot:
        for p in decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
        for p in generator.parameters():
            if p.dim() > 1:
                xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))

    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        decoder.half()
    
    return decoder, generator



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
        for (name, field), (checkpoint_name, checkpoint_field) in zip(
            multifield, checkpoint_multifield
        ):
            new_tokens = []
            for i, tok in enumerate(field.vocab.itos):
                if tok in checkpoint_field.vocab.stoi:
                    old_i = checkpoint_field.vocab.stoi[tok]
                    model.state_dict()[emb_name][i] = checkpoint["model"][
                        emb_name
                    ][old_i]
                    if field_name == "tgt":
                        generator.state_dict()["0.weight"][i] = checkpoint[
                            "generator"
                        ]["0.weight"][old_i]
                        generator.state_dict()["0.bias"][i] = checkpoint[
                            "generator"
                        ]["0.bias"][old_i]
                else:
                    # Just for debugging purposes
                    new_tokens.append(tok)
            logger.info("%s: %d new tokens" % (name, len(new_tokens)))
        # Remove old vocabulary associated embeddings
        del checkpoint["model"][emb_name]
    del checkpoint["generator"]["0.weight"], checkpoint["generator"]["0.bias"]


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
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
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")

    model = build_task_specific_model(model_opt, fields)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = model.decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator.linear.weight = model.decoder.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is None or model_opt.update_vocab:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility

        if model_opt.update_vocab:
            # Update model embeddings with those from the checkpoint
            # after initialization
            use_embeddings_from_checkpoint(fields, model, generator,
                                           checkpoint)

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
    return model

#def build_base_model_langspec(model_opt, fields, gpu, checkpoint=None, nodeRank=None, deviceRank=None):
def build_base_model_langspec(model_opt, Fields_dict, gpu, checkpoint=None, nodeRank=None, deviceRank=None):
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
    """
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    """
    logger.info("MODEL BUILDER")
    device = torch.device("cuda")
    logger.info(device)
#    model, generators_md, langs = build_task_specific_model(model_opt, fields, device, checkpoint, nodeRank, deviceRank)
    model, generators_md = build_task_specific_model(model_opt, Fields_dict, device, checkpoint, nodeRank, deviceRank)

    model.generator = generators_md
    model.to(device)

    #model.half()

    # Load the model states from checkpoint or initialize them.
#    if checkpoint is None or model_opt.update_vocab:
#        if hasattr(model, "encoder") and hasattr(model.encoder, "embeddings"):
#            model.encoder.embeddings.load_pretrained_vectors(
#                model_opt.pre_word_vecs_enc)
#        if hasattr(model.decoder, 'embeddings'):
#            model.decoder.embeddings.load_pretrained_vectors(
#                model_opt.pre_word_vecs_dec)

#    model.generator = generator
    return model, generators_md

#def build_model(model_opt, opt, fields, checkpoint, nodeRank, deviceRank):
def build_model(model_opt, opt, Fields_dict, checkpoint, nodeRank, deviceRank):
    logger.info('Building model...')
#    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model, generators_md = build_base_model_langspec(model_opt, Fields_dict, use_gpu(opt), checkpoint, nodeRank, deviceRank)
#    model, generators_md, langs = build_base_model_langspec(model_opt, fields, use_gpu(opt), checkpoint, nodeRank, deviceRank)
    logger.info(model)
    return model, generators_md
