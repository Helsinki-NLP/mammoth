#!/usr/bin/env python
import argparse
import torch


def average_models(model_files, fp32=False):
    vocab = None
    opts = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        m = torch.load(model_file, map_location='cpu')
        model_weights = m['model']
        generator_weights = m['generator']

        if fp32:
            for k, v in model_weights.items():
                model_weights[k] = v.float()
            for k, v in generator_weights.items():
                generator_weights[k] = v.float()

        if i == 0:
            vocab, opts = m['vocab'], m['opts']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)

    final = {"vocab": vocab, "opts": opts, "optim": None, "generator": avg_generator, "model": avg_model}
    return final


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-models", "-m", nargs="+", required=True, help="List of models")
    parser.add_argument("-output", "-o", required=True, help="Output file")
    parser.add_argument("-fp32", "-f", action="store_true", help="Cast params to float32")
    opts = parser.parse_args()

    final = average_models(opts.models, opts.fp32)
    torch.save(final, opts.output)


if __name__ == "__main__":
    main()
