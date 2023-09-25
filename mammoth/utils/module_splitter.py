from collections import OrderedDict
from typing import Dict


def _combine_ordered_dicts(input_dicts: Dict[str, OrderedDict]) -> OrderedDict:
    result = []
    for prefix, input_dict in input_dicts.items():
        for key, item in input_dict.items():
            result.append((f'{prefix}{key}', item))
    return OrderedDict(result)


def explode_model(full_ab_model):
    encoder = full_ab_model["whole_model"].encoder
    decoder = full_ab_model["whole_model"].decoder

    modules = {}

    # embeddings
    for embedding_key, embeddings in encoder.embeddings.items():
        lang = embedding_key.replace('embeddings_', '')
        key = f'src_embeddings_{lang}'
        modules[key] = embeddings.state_dict()
    for embedding_key, embeddings in decoder.embeddings.items():
        lang = embedding_key.replace('embeddings_', '')
        key = f'tgt_embeddings_{lang}'
        modules[key] = embeddings.state_dict()

    # encoders
    for layer_stack_idx, layer_stack_dict in enumerate(encoder.encoders):
        for layer_stack_key, layer_stack in layer_stack_dict.items():
            # the xcoder itself
            key = f'encoder_{layer_stack_idx}_{layer_stack_key}'
            modules[key] = layer_stack.state_dict(include_adapters=False)

            # the adapters for this xcoder
            for adapter_key, adapter in layer_stack.adapters.items():
                adapter_key = adapter_key.replace('adapter_', '')
                key = f'encoder_adapter_{layer_stack_idx}_{layer_stack_key}_{adapter_key}'
                modules[key] = adapter.state_dict()

    # decoders
    for layer_stack_idx, layer_stack_dict in enumerate(decoder.decoders):
        for layer_stack_key, layer_stack in layer_stack_dict.items():
            # the xcoder itself
            key = f'decoder_{layer_stack_idx}_{layer_stack_key}'
            modules[key] = layer_stack.state_dict(include_adapters=False)

            # the adapters for this xcoder
            for adapter_key, adapter in layer_stack.adapters.items():
                adapter_key = adapter_key.replace('adapter_', '')
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
        "opts": full_ab_model["opts"],
        "optim": full_ab_model["optim"],
    }

    return modules, model_frame
