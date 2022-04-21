from collections import OrderedDict
from copy import deepcopy
from typing import List

from torch.nn.modules.container import ModuleList


def _extract_matching_keys(
    input_dict: OrderedDict,
    partial_key: str,
    matching_method=lambda k, pk: k.startswith(pk),
    transform_key=lambda x: x,
) -> OrderedDict:
    # TODO: improve typing
    return OrderedDict(
        (transform_key(key), value)
        for key, value in input_dict.items()
        if matching_method(key, partial_key)
    )


def _combine_ordered_dicts(input_dicts: List[OrderedDict]) -> OrderedDict:
    out = OrderedDict()
    for input_dict in input_dicts:
        out = OrderedDict(list(out.items()) + list(input_dict.items()))
    return out


def _compose_vocab(
    frame_vocab: OrderedDict, src_lang: str, tgt_lang: str
) -> OrderedDict:

    src_tgt_langs = "{}-{}".format(src_lang, tgt_lang)
    out = OrderedDict(
        [
            (
                src_tgt_langs,
                {
                    "src": list(
                        _extract_matching_keys(
                            frame_vocab, "{}-".format(src_lang)
                        ).values()
                    )[0]["src"],
                    "tgt": list(
                        _extract_matching_keys(
                            frame_vocab,
                            "-{}".format(tgt_lang),
                            matching_method=lambda k, pk: pk in k,
                        ).values()
                    )[0]["tgt"],
                    "indices": list(frame_vocab.values())[0]["indices"],
                },
            )
        ]
    )

    return out


def explode_model(full_ab_model):

    enc_modules = list()
    dec_modules = list()
    ab_module = OrderedDict()
    gen_modules = list()

    enc_lang_to_ids = full_ab_model["whole_model"].encoder_ids
    dec_lang_to_ids = full_ab_model["whole_model"].decoder_ids
    enc_ids_to_lang = {index: lang for lang, index in enc_lang_to_ids.items()}
    dec_ids_to_lang = {index: lang for lang, index in dec_lang_to_ids.items()}

    # extract encoders
    for i, encoder in enumerate(full_ab_model["whole_model"].encoders):
        lang = enc_ids_to_lang[i]
        enc_model = OrderedDict()
        enc_model["whole_model"] = OrderedDict(
            [
                ("encoder", encoder),
                (
                    "encoder_ids",  # TODO: remove if not used in build_bilingual_model (but can be used for a multilingual model)
                    {lang: i},
                ),
                (
                    "encoder_fams",  # TODO: remove if not used in build_bilingual_model (but can be used for a multilingual model)
                    {lang: full_ab_model["whole_model"].encoder_fams[lang]},
                ),
                (
                    "encoder_grps",  # TODO: remove if not used in build_bilingual_model (but can be used for a multilingual model)
                    {lang: full_ab_model["whole_model"].encoder_grps[lang]},
                ),
                (
                    "_modules",
                    OrderedDict(
                        [
                            (
                                "encoder",
                                OrderedDict(
                                    [
                                        (
                                            "module",
                                            full_ab_model["whole_model"]._modules[
                                                "encoders"
                                            ][i],
                                        ),
                                        (
                                            "_modules",
                                            full_ab_model["whole_model"]
                                            ._modules["encoders"]
                                            ._modules[str(i)],
                                        ),
                                    ]
                                ),
                            )
                        ]
                    ),
                ),
            ]
        )
        if lang in full_ab_model["whole_model"].tgt_vocabs.keys():
            enc_model["whole_model"]["tgt_vocabs"] = {lang: full_ab_model["whole_model"].tgt_vocabs[lang]}
        enc_model["model"] = _extract_matching_keys(
            full_ab_model["model"],
            "encoders.{}.".format(i),
            transform_key=lambda x: x.replace("encoders.{}.".format(i), "encoders.0."),
        )
        enc_model["model"]._metadata = _extract_matching_keys(
            full_ab_model["model"]._metadata,
            "encoders.{}.".format(i),
            transform_key=lambda x: x.replace("encoders.{}.".format(i), "encoders.0."),
        )
        enc_modules.append(enc_model)

    # extract decoders
    for i, decoder in enumerate(full_ab_model["whole_model"].decoders):
        lang = dec_ids_to_lang[i]
        dec_model = OrderedDict()
        dec_model["whole_model"] = OrderedDict(
            [
                ("decoder", decoder),
                (
                    "decoder_ids",  # TODO: remove if not used in build_bilingual_model (but can be used for a multilingual model)
                    {lang: i},
                ),
                (
                    "decoder_types",
                    {lang: full_ab_model["whole_model"].decoder_types[lang]},
                ),
                ("tgt_vocabs", {lang: full_ab_model["whole_model"].tgt_vocabs[lang]}),
                (
                    "_modules",
                    OrderedDict(
                        [
                            (
                                "decoder",
                                OrderedDict(
                                    [
                                        (
                                            "module",
                                            full_ab_model["whole_model"]._modules[
                                                "decoders"
                                            ][i],
                                        ),
                                        (
                                            "_modules",
                                            full_ab_model["whole_model"]
                                            ._modules["decoders"]
                                            ._modules[str(i)],
                                        ),
                                    ]
                                ),
                            )
                        ]
                    ),
                ),
            ]
        )
        dec_model["model"] = _extract_matching_keys(
            full_ab_model["model"],
            "decoders.{}.".format(i),
            transform_key=lambda x: x.replace("decoders.{}.".format(i), "decoders.0."),
        )
        dec_model["model"]._metadata = _extract_matching_keys(
            full_ab_model["model"]._metadata,
            "decoders.{}.".format(i),
            transform_key=lambda x: x.replace("decoders.{}.".format(i), "decoders.0."),
        )
        dec_modules.append(dec_model)

    # extract attention bridge
    ab_module["whole_model"] = full_ab_model["whole_model"].attention_bridge
    ab_module["model"] = _extract_matching_keys(
        full_ab_model["model"], "attention_bridge."
    )
    ab_module["model"]._metadata = _extract_matching_keys(
        full_ab_model["model"]._metadata, "attention_bridge."
    )

    # extract generators
    for i, generator in enumerate(full_ab_model["whole_model"].generators):
        gen_module = OrderedDict()
        gen_module["whole_model"] = OrderedDict(
            [
                ("generator", generator),
                (
                    "_modules",
                    OrderedDict(
                        [
                            (
                                "generator",
                                OrderedDict(
                                    [
                                        (
                                            "module",
                                            full_ab_model["whole_model"]._modules[
                                                "generators"
                                            ][i],
                                        ),
                                        (
                                            "_modules",
                                            full_ab_model["whole_model"]
                                            ._modules["generators"]
                                            ._modules[str(i)],
                                        ),
                                    ]
                                ),
                            )
                        ]
                    ),
                ),
            ]
        )  # using an OrderedDict for consistency with encoders and decoders
        gen_module["model"] = _extract_matching_keys(
            full_ab_model["model"],
            "generators.{}".format(i),
            transform_key=lambda x: x.replace(
                "generators.{}.".format(i), "generators.0."
            ),
        )
        gen_module["model"]._metadata = _extract_matching_keys(
            full_ab_model["model"]._metadata,
            "generators.{}".format(i),
            transform_key=lambda x: x.replace(
                "generators.{}.".format(i), "generators.0."
            ),
        )
        gen_modules.append(gen_module)

    # stuff necessary to build bilingual models combining modules
    # TODO: remove the stuff that is already stored in the other modules!
    model_frame = {
        "vocab": full_ab_model["vocab"],
        "opt": full_ab_model["opt"],
        "optim": full_ab_model["optim"],
        "whole_model": full_ab_model["whole_model"],
    }

    return enc_modules, dec_modules, ab_module, gen_modules, model_frame


def build_bilingual_model(
        src_lang: str,
        tgt_lang: str,
        enc_module: OrderedDict,
        dec_module: OrderedDict,
        ab_module: OrderedDict,
        gen_module: OrderedDict,
        model_frame: dict,
):

    src_langs_to_id = model_frame["whole_model"].encoder_ids
    tgt_langs_to_id = model_frame["whole_model"].decoder_ids

    output_model = deepcopy(model_frame)
    output_model["model"] = _combine_ordered_dicts(
        [
            enc_module["model"],
            dec_module["model"],
            ab_module["model"],
            gen_module["model"],
        ]
    )
    output_model["model"]._metadata = _combine_ordered_dicts(
        [
            enc_module["model"]._metadata,
            dec_module["model"]._metadata,
            ab_module["model"]._metadata,
            gen_module["model"]._metadata,
        ]
    )
    output_model["vocab"] = _compose_vocab(model_frame["vocab"], src_lang, tgt_lang)

    output_model["whole_model"].encoders = ModuleList(
        [enc_module["whole_model"]["encoder"]]
    )
    output_model["whole_model"].decoders = ModuleList(
        [dec_module["whole_model"]["decoder"]]
    )
    output_model["whole_model"].generators = ModuleList(
        [gen_module["whole_model"]["generator"]]
    )

    output_model["whole_model"].decoder_ids = {tgt_lang: 0}
    output_model["whole_model"].encoder_ids = {src_lang: 0}
    output_model["whole_model"].decoder_types = dec_module["whole_model"][
        "decoder_types"
    ]
    output_model["whole_model"].encoder_fams = {src_lang: 0}
    output_model["whole_model"].encoder_grps = {src_lang: 0}
    output_model["whole_model"].tgt_vocabs = {
        **enc_module["whole_model"]["tgt_vocabs"],
        **dec_module["whole_model"]["tgt_vocabs"],
    }
    output_model["whole_model"]._modules["encoders"]._modules = OrderedDict(
        [("0", enc_module["whole_model"]["_modules"]["encoder"]["_modules"])]
    )
    output_model["whole_model"]._modules["decoders"]._modules = OrderedDict(
        [("0", dec_module["whole_model"]["_modules"]["decoder"]["_modules"])]
    )
    output_model["whole_model"]._modules["generators"]._modules = OrderedDict(
        [("0", gen_module["whole_model"]["_modules"]["generator"]["_modules"])]
    )

    return output_model