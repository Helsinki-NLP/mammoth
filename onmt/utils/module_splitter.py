from collections import OrderedDict
from typing import List


def _combine_ordered_dicts(input_dicts: List[OrderedDict]) -> OrderedDict:
    out = OrderedDict()
    for input_dict in input_dicts:
        out = OrderedDict(list(out.items()) + list(input_dict.items()))
    return out


def explode_model(full_ab_model):

    enc_modules = list()
    dec_modules = list()
    ab_module = OrderedDict()
    gen_modules = list()

    # extract encoders
    for k, encoder in full_ab_model["whole_model"].encoder.items():
        enc_model = {'model': encoder.state_dict()}
        enc_modules.append(enc_model)

    # extract decoders
    for k, decoder in full_ab_model["whole_model"].decoder.items():
        dec_model = {'model': decoder.state_dict()}
        dec_modules.append(dec_model)

    # extract attention bridge
    ab_module["model"] = full_ab_model['whole_model'].attention_bridge.state_dict()

    # extract generators
    for k, generator in full_ab_model["whole_model"].generator.items():
        gen_module = {'model': generator.state_dict()}
        gen_modules.append(gen_module)

    # stuff necessary to build bilingual models combining modules
    model_frame = {
        "vocab": full_ab_model["vocab"],
        "opt": full_ab_model["opt"],
        "optim": full_ab_model["optim"],
        # "whole_model": full_ab_model["whole_model"],
    }

    return enc_modules, dec_modules, ab_module, gen_modules, model_frame


def create_bilingual_statedict(
    src_lang: str,
    tgt_lang: str,
    enc_module: OrderedDict,
    dec_module: OrderedDict,
    ab_module: OrderedDict,
    gen_module: OrderedDict,
):

    enc = {f'encoder.encoder{src_lang}.{k}': v for k, v in enc_module["model"].items()}
    dec = {f'decoder.decoder{tgt_lang}.{k}': v for k, v in dec_module["model"].items()}
    ab = {f'attention_bridge.{k}': v for k, v in ab_module["model"].items()}
    gen = {f'generator.generator{tgt_lang}.{k}': v for k, v in gen_module["model"].items()}

    output_model = _combine_ordered_dicts([enc, dec, ab, gen])

    return output_model
