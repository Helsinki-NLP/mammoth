from collections import OrderedDict
from typing import List


def _combine_ordered_dicts(input_dicts: List[OrderedDict]) -> OrderedDict:
    out = OrderedDict()
    for input_dict in input_dicts:
        out = OrderedDict(list(out.items()) + list(input_dict.items()))
    return out


def explode_model(full_ab_model):
    encoder = full_ab_model["whole_model"].encoder
    decoder = full_ab_model["whole_model"].decoder

    modules = {}

    # embeddings
    for embedding_key, embeddings in encoder.embeddings.items():
        lang = embedding_key.replace('embeddings', '')
        key = f'src_embeddings_{lang}'
        modules[key] = embeddings.state_dict()
    for embedding_key, embeddings in decoder.embeddings.items():
        lang = embedding_key.replace('embeddings', '')
        key = f'tgt_embeddings_{lang}'
        modules[key] = embeddings.state_dict()

    # encoders
    for layer_stack_idx, layer_stack_dict in enumerate(encoder.encoders):
        for layer_stack_key, layer_stack in layer_stack_dict.items():
            # the xcoder itself
            key = f'encoder_{layer_stack_idx}_{layer_stack_key}'
            modules[key] = layer_stack.state_dict()

            # the adapters for this xcoder
            for adapter_key, adapter in layer_stack.adapters.items():
                key = f'encoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_key}'
                modules[key] = adapter.state_dict()

    # decoders
    for layer_stack_idx, layer_stack_dict in enumerate(decoder.decoders):
        for layer_stack_key, layer_stack in layer_stack_dict.items():
            # the xcoder itself
            key = f'decoder_{layer_stack_idx}_{layer_stack_key}'
            modules[key] = layer_stack.state_dict()

            # the adapters for this xcoder
            for adapter_key, adapter in layer_stack.adapters.items():
                key = f'decoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_key}'
                modules[key] = adapter.state_dict()

    # generators
    for generator_key, generator in full_ab_model["whole_model"].generator.items():
        modules[generator_key] = generator.state_dict()

    # attention bridge
    modules['attention_bridge'] = full_ab_model['whole_model'].attention_bridge.state_dict()

    # stuff necessary to build bilingual models combining modules
    model_frame = {
        "vocab": full_ab_model["vocab"],
        "opt": full_ab_model["opt"],
        "optim": full_ab_model["optim"],
    }

    return modules, model_frame


def create_bilingual_statedict(
    enc_id: str,
    dec_id: str,
    tgt_lang: str,
    enc_module: OrderedDict,
    dec_module: OrderedDict,
    ab_module: OrderedDict,
    gen_module: OrderedDict,
):

    enc = {f'encoder.encoder{enc_id}.{k}': v for k, v in enc_module["model"].items()}
    dec = {f'decoder.decoder{dec_id}.{k}': v for k, v in dec_module["model"].items()}
    ab = {f'attention_bridge.{k}': v for k, v in ab_module["model"].items()}
    gen = {f'generator.generator{tgt_lang}.{k}': v for k, v in gen_module["model"].items()}

    output_model = _combine_ordered_dicts([enc, dec, ab, gen])

    return output_model
