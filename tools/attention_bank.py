import argparse
import collections
import itertools
import pathlib

import numpy as np
import torch
import tqdm

from mammoth.inputters.dataset import ParallelCorpus
from mammoth.inputters.dataloader import build_dataloader
from mammoth.model_builder import load_test_multitask_model
from mammoth.opts import build_bilingual_model, _add_dynamic_transform_opts
from mammoth.transforms import get_transforms_cls, make_transforms, TransformPipe
from mammoth.utils.parse import ArgumentParser


def get_opts():
    """parse command line options"""

    parser = argparse.ArgumentParser()
    # translate_opts(parser, dynamic=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=torch.device, default=torch.device('cpu'))
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lang_pair', type=str, required=False)
    build_bilingual_model(parser)
    parser.add_argument('--transforms', type=str, nargs='*', default=[])
    _add_dynamic_transform_opts(parser)

    subparsers = parser.add_subparsers(dest='command')

    subparser = subparsers.add_parser('extract')
    subparser.add_argument('--src_sentences', type=pathlib.Path)
    subparser.add_argument('--dump_file', type=pathlib.Path)

    subparser = subparsers.add_parser('estimate')
    subparser.add_argument('--src_sentences', type=pathlib.Path, required=True)
    subparser.add_argument('--param_save_file', type=pathlib.Path, required=True)
    # subparser.add_argument('--do_generate', action='store_true')
    # subparser.add_argument('--top_k', type=int, default=None)

    subparser = subparsers.add_parser('classify')
    subparser.add_argument('--train_sentences', type=pathlib.Path, nargs='+')
    subparser.add_argument('--train_labels', type=pathlib.Path)
    subparser.add_argument('--test_sentences', type=pathlib.Path, nargs='+')
    subparser.add_argument('--test_labels', type=pathlib.Path)
    subparser.add_argument('--n_runs', type=int, default=10)

    opts = parser.parse_args()

    return opts


def _extract(sentences_file, model, vocabs_dict, transforms, enc_id, batch_size=100, device='cpu'):
    """sub-routine to embed file"""
    corpus = ParallelCorpus(
        sentences_file,
        None,  # no target
        vocabs_dict['src'],
        vocabs_dict['tgt'],
        transforms=transforms,
    ).to(device)

    batches = build_dataloader(
        corpus,
        batch_size=batch_size,
        batch_type='sents',
        cycle=False,
    )

    for batch in batches:
        src, src_lengths = batch.src
        enc_states, memory_bank, src_lengths, mask = model.encoder[f"encoder{enc_id}"](
            src.to(device), src_lengths.to(device)
        )
        memory_bank, alphas = model.attention_bridge(memory_bank, mask)
        yield (memory_bank, src_lengths)


def extract(opts, vocabs_dict, model, model_opts, transforms):
    """Compute representations drawn from the encoder and save them to file."""
    sentence_reps = []
    for src, src_length in _extract(
        opts.src_sentences,
        model,
        vocabs_dict,
        transforms,
        opts.enc_id,
        batch_size=opts.batch_size,
        device=opts.device,
    ):
        for sentence, length in zip(src.cpu().unbind(1), src_length):
            sentence_reps.append(sentence[:length])
    torch.save(sentence_reps, opts.dump_file)


def estimate(opts, vocabs_dict, model, model_opts, transforms):
    """Estimate the matrix-variate distribution of representations drawn from the encoder."""
    try:
        import sklearn.covariance
    except ImportError:
        raise RuntimeError('please install scikit-learn')

    assert model.attention_bridge.is_fixed_length, "Can't estimate matrix-variate distribution with varying shapes"
    print('Extracting sentences...')
    sentence_reps, _ = zip(
        * _extract(
            opts.src_sentences,
            model,
            vocabs_dict,
            transforms,
            opts.enc_id,
            batch_size=opts.batch_size,
            device=opts.device,
        )
    )
    sentence_reps = torch.cat(sentence_reps, dim=1).transpose(0, 1)
    b, fixed_len, feats = sentence_reps.size()
    sentence_reps = sentence_reps.contiguous().view(b, fixed_len * feats).cpu().numpy()
    print('Estimating covariance...')
    cov_obj = sklearn.covariance.EmpiricalCovariance()
    cov_obj.fit(sentence_reps)
    loc = torch.from_numpy(cov_obj.location_)
    cov_mat = torch.from_numpy(cov_obj.covariance_)
    print('Saving params...')
    torch.save([loc, cov_mat, fixed_len, feats], opts.param_save_file)
    # TODO: need to implement some way of using this in translate.py
    # how to use:
    # loc, cov_mat, fixed_len, feats = torch.load(opts.param_save_file)
    # dist = torch.distributions.MultivariateNormal(loc, covariance_matrix=cov_mat,)
    # def sampling_fn():
    #     return dist.sample().view(fixed_len, 1, feats)
    # return sampling_fn


def classify(opts, vocabs_dict, model, model_opts, transforms):
    """Learn a simple SGD classifier using representations drawn from the encoder."""
    try:
        import sklearn.linear_model
    except ImportError:
        raise RuntimeError('please install scikit-learn')

    label2idx = collections.defaultdict(itertools.count().__next__)

    X_trains = []
    for train_file in tqdm.tqdm(opts.train_sentences, desc='train files'):
        embedded = []
        if train_file.suffix == '.pt':
            print(f"Assuming {train_file} corresponds to pre-extracted embeddings; change suffix if it's not.")
            for sentence_rep in torch.load(train_file):
                embedded.append(sentence_rep.sum(0, keepdim=True))
        else:
            for src, src_lengths in tqdm.tqdm(
                _extract(
                    train_file,
                    model,
                    vocabs_dict,
                    transforms,
                    opts.enc_id,
                    batch_size=opts.batch_size,
                    device=opts.device,
                ),
                leave=False,
            ):
                mask = torch.arange(src.size(0), device=opts.device).unsqueeze(1) >= src_lengths.unsqueeze(0)
                embedded.append(src.masked_fill(mask.unsqueeze(-1), 0.).sum(0).cpu())
        X_trains.append(torch.cat(embedded, dim=0))
    X_train = torch.cat(X_trains, dim=-1).numpy()

    with open(opts.train_labels, 'r') as istr:
        istr = map(str.strip, istr)
        istr = map(label2idx.__getitem__, istr)
        y_train = np.array(list(istr))

    label2idx = dict(label2idx)

    X_test = []
    for test_file in tqdm.tqdm(opts.test_sentences, desc='test files'):
        embedded = []
        if test_file.suffix == '.pt':
            print(f"Assuming {test_file} corresponds to pre-extracted embeddings; change suffix if it's not.")
            for sentence_rep in torch.load(test_file):
                embedded.append(sentence_rep.sum(0, keepdim=True))
        else:
            for src, src_lengths in tqdm.tqdm(
                _extract(
                    test_file,
                    model,
                    vocabs_dict,
                    transforms,
                    opts.enc_id,
                    batch_size=opts.batch_size,
                    device=opts.device,
                ),
                leave=False,
            ):
                mask = torch.arange(src.size(0), device=opts.device).unsqueeze(1) >= src_lengths.unsqueeze(0)
                embedded.append(src.masked_fill(mask.unsqueeze(-1), 0.).sum(0).cpu())
        X_test.append(torch.cat(embedded, dim=0))
    X_test = torch.cat(X_test, dim=-1).numpy()

    with open(opts.test_labels, 'r') as istr:
        istr = map(str.strip, istr)
        istr = map(label2idx.__getitem__, istr)
        y_test = np.array(list(istr))

    print('Training classifiers...')
    scores = []
    for _ in range(opts.n_runs):
        model = sklearn.linear_model.SGDClassifier(max_iter=100_000, n_jobs=-1, early_stopping=True)
        model.fit(X_train, y_train)
        scores.append(f'{model.score(X_test, y_test) * 100:.4f}%')

    print(f'Scores: {" ".join(scores)}')


@torch.no_grad()
def main():
    """Main entry point"""
    opts = get_opts()
    ArgumentParser._get_all_transform_translate(opts)
    ArgumentParser._validate_transforms_opts(opts)
    # ArgumentParser.validate_translate_opts_dynamic(opts)
    opts.enc_id = opts.enc_id or opts.src_lang

    vocabs_dict, model, model_opts = load_test_multitask_model(opts, opts.model)
    command_fn = {
        fn.__name__: fn for fn in
        [extract, estimate, classify]
    }[opts.command]

    # Build transforms
    transforms_cls = get_transforms_cls(opts._all_transform)
    transforms = make_transforms(opts, transforms_cls, vocabs_dict)
    data_transform = [
        transforms[name] for name in opts.transforms if name in transforms
    ]
    transform = TransformPipe.build_from(data_transform)

    command_fn(opts, vocabs_dict, model.to(opts.device), model_opts, transform)


if __name__ == '__main__':
    main()
