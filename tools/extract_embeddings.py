import argparse

import torch

import mammoth
import mammoth.model_builder

from mammoth.utils.parse import ArgumentParser
import mammoth.opts

from mammoth.utils.misc import use_gpu
from mammoth.utils.logging import init_logger, logger

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True, help='Path to model .pt file')
parser.add_argument('-output_dir', default='.', help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1, help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'wb') as file:
        for i in range(min(len(embeddings), len(dict.itos))):
            str = dict.itos[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + b"\n")


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    mammoth.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opts = parser.parse_args()
    opts.cuda = opts.gpu > -1
    if opts.cuda:
        torch.cuda.set_device(opts.gpu)

    # Add in default model arguments, possibly added since training.
    checkpoint = torch.load(opts.model, map_location=lambda storage, loc: storage)
    model_opts = checkpoint['opts']

    fields = checkpoint['vocab']
    src_dict = fields['src'].base_field.vocab  # assumes src is text
    tgt_dict = fields['tgt'].base_field.vocab

    model_opts = checkpoint['opts']
    for arg in dummy_opt.__dict__:
        if arg not in model_opts:
            model_opts.__dict__[arg] = dummy_opt.__dict__[arg]

    # build_base_model expects updated and validated opts
    ArgumentParser.update_model_opts(model_opts)
    ArgumentParser.validate_model_opts(model_opts)

    model = mammoth.model_builder.build_base_model(model_opts, fields, use_gpu(opts), checkpoint)
    encoder = model.encoder  # no encoder for LM task
    decoder = model.decoder

    encoder_embeddings = encoder.embeddings.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.embeddings.word_lut.weight.data.tolist()

    logger.info("Writing source embeddings")
    write_embeddings(opts.output_dir + "/src_embeddings.txt", src_dict, encoder_embeddings)

    logger.info("Writing target embeddings")
    write_embeddings(opts.output_dir + "/tgt_embeddings.txt", tgt_dict, decoder_embeddings)

    logger.info('... done.')
    logger.info('Converting model...')


if __name__ == "__main__":
    init_logger('extract_embeddings.log')
    main()
