#!/usr/bin/env python
""" Translator Class and builder """
import codecs
import os
import time
import numpy as np
import warnings
from itertools import count, zip_longest
from einops import rearrange

import torch

# from mammoth.inputters.text_dataset import InferenceDataIterator
from mammoth.constants import DefaultTokens
from mammoth.inputters.dataloader import build_dataloader
from mammoth.inputters.dataset import ParallelCorpus
from mammoth.model_builder import build_model
from mammoth.translate.beam_search import BeamSearch, GNMTGlobalScorer
from mammoth.translate.greedy_search import GreedySearch
from mammoth.translate.translation import TranslationBuilder
from mammoth.utils.alignment import build_align_pharaoh
from mammoth.utils.misc import set_random_seed, report_matrix, use_gpu
from mammoth.utils.model_saver import load_frame_checkpoint, load_parameters_from_checkpoint
from mammoth.utils.parse import ArgumentParser


def build_translator(opts, task_queue_manager, task, report_score=True, logger=None, out_file=None):
    if out_file is None:
        outdir = os.path.dirname(opts.output)
        if outdir and not os.path.isdir(outdir):
            warnings.warn(f'output file directory "{outdir}" does not exist... creating it.')
            os.makedirs(os.path.dirname(opts.output), exist_ok=True)
        out_file = codecs.open(opts.output, "w+", "utf-8")

    # TODO: reimplement ensemble decoding
    load_model_for_translation_func = load_model_for_translation
    if logger:
        logger.info(str(task))
    model_path = None
    vocabs, model, model_opts = load_model_for_translation_func(
        opts=opts,
        task_queue_manager=task_queue_manager,
        task=task,
        model_path=model_path,
    )

    scorer = GNMTGlobalScorer.from_opts(opts)

    translator = Translator.from_opts(
        model,
        vocabs,
        opts,
        model_opts,
        global_scorer=scorer,
        out_file=out_file,
        report_align=opts.report_align,
        report_score=report_score,
        logger=logger,
        task=task,
    )
    return translator


def load_model_for_translation(opts, task_queue_manager, task=None, model_path=None):
    if task is None:
        raise ValueError('Must set task')
    if model_path is None:
        model_path = opts.models[0]

    # Load only the frame
    frame, frame_checkpoint_path = load_frame_checkpoint(checkpoint_path=model_path)

    vocabs_dict = {
        ('src', task.src_lang): frame["vocab"].get(('src', task.src_lang)),
        ('tgt', task.tgt_lang): frame["vocab"].get(('tgt', task.tgt_lang)),
        'src': frame["vocab"].get(('src', task.src_lang)),
        'tgt': frame["vocab"].get(('tgt', task.tgt_lang)),
    }
    print(f'vocabs_dict {vocabs_dict}')
    print(f'my compontents {task_queue_manager.get_my_distributed_components()}')

    model_opts = ArgumentParser.checkpoint_model_opts(frame['opts'])

    model = build_model(
        model_opts,
        opts,
        vocabs_dict,
        task_queue_manager,
        single_task=task.corpus_id,
    )

    load_parameters_from_checkpoint(
        frame_checkpoint_path,
        model,
        optim=None,
        task_queue_manager=task_queue_manager,
        reset_optim=True,
    )

    device = torch.device("cuda" if use_gpu(opts) else "cpu")
    model.to(device)
    model.eval()

    return vocabs_dict, model, model_opts


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Inference(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (mammoth.modules.NMTModel): NMT model to use for translation
        vocabs (dict[str, mammoth.inputters.Vocab]): A dict
            mapping each side to its Vocab.
        src_file_path (str): Source file to read.
        tgt_reader (src): Target file, if necessary.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`mammoth.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`mammoth.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`mammoth.translate.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`mammoth.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`mammoth.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`mammoth.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        tgt_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        global_scorer (mammoth.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        vocabs,
        src_file_path,
        tgt_file_path=None,
        gpu=-1,
        n_best=1,
        min_length=0,
        max_length=100,
        ratio=0.0,
        beam_size=30,
        random_sampling_topk=0,
        random_sampling_topp=0.0,
        random_sampling_temp=1.0,
        stepwise_penalty=None,
        dump_beam=False,
        block_ngram_repeat=0,
        ignore_when_blocking=frozenset(),
        replace_unk=False,
        ban_unk_token=False,
        tgt_prefix=False,
        phrase_table="",
        data_type="text",
        verbose=False,
        report_time=False,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
        seed=-1,
        task=None,
    ):
        assert task is not None
        self.task = task
        if logger:
            logger.info(f'task {task}')

        self.model = model
        self.vocabs = vocabs
        tgt_vocab = dict(self.vocabs)[("tgt", task.tgt_lang)]
        self._tgt_vocab = tgt_vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[DefaultTokens.EOS]
        self._tgt_pad_idx = self._tgt_vocab.stoi[DefaultTokens.PAD]
        self._tgt_bos_idx = self._tgt_vocab.stoi[DefaultTokens.BOS]
        self._tgt_unk_idx = self._tgt_vocab.stoi[DefaultTokens.UNK]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._device = torch.device("cuda", self._gpu) if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.min_length = min_length
        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        # self.src_reader = src_reader
        # self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError("replace_unk requires an attentional decoder.")
        self.tgt_prefix = tgt_prefix
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_time = report_time

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and not self.model.decoder.attentional:
            raise ValueError("Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_align = report_align
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opts(
        cls,
        model,
        vocabs,
        opts,
        model_opts,
        global_scorer=None,
        out_file=None,
        report_align=False,
        report_score=True,
        logger=None,
        task=None,
    ):
        """Alternate constructor.

        Args:
            model (mammoth.modules.NMTModel): See :func:`__init__()`.
            vocabs (dict[str, mammoth.inputters.Vocab]): See
                :func:`__init__()`.
            opts (argparse.Namespace): Command line options
            model_opts (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (mammoth.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_align (bool) : See :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """
        assert task is not None
        # TODO: maybe add dynamic part
        # cls.validate_task(model_opts.model_task)

        return cls(
            model,
            vocabs,
            opts.src,
            tgt_file_path=opts.tgt,
            gpu=opts.gpu_rank,
            n_best=opts.n_best,
            min_length=opts.min_length,
            max_length=opts.max_length,
            ratio=opts.ratio,
            beam_size=opts.beam_size,
            random_sampling_topk=opts.random_sampling_topk,
            random_sampling_topp=opts.random_sampling_topp,
            random_sampling_temp=opts.random_sampling_temp,
            stepwise_penalty=opts.stepwise_penalty,
            dump_beam=opts.dump_beam,
            block_ngram_repeat=opts.block_ngram_repeat,
            ignore_when_blocking=set(opts.ignore_when_blocking),
            replace_unk=opts.replace_unk,
            ban_unk_token=opts.ban_unk_token,
            tgt_prefix=task.corpus_opts.get('tgt_prefix', None),
            phrase_table=opts.phrase_table,
            data_type=opts.data_type,
            verbose=opts.verbose,
            report_time=opts.report_time,
            global_scorer=global_scorer,
            out_file=out_file,
            report_align=report_align,
            report_score=report_score,
            logger=logger,
            seed=opts.seed,
            task=task,
        )

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(
        self,
        active_decoder,
        batch,
        encoder_output,
        src_mask,
        src_vocabs,
        batch_size,
    ):
        if batch.tgt is not None:
            gs = self._score_target(
                active_decoder,
                batch,
                encoder_output,
                src_mask,
                src_vocabs,
                batch_size,
            )
        else:
            gs = [0] * batch_size
        return gs

    def translate_dynamic(
        self,
        src,
        transform,
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table=""
    ):

        if batch_size is None:
            raise ValueError("batch_size must be set")

        #
        # data_iter = InferenceDataIterator(src, tgt, src_feats, transform)
        #
        # data = inputters.DynamicDataset(
        #     self.fields,
        #     data=data_iter,
        #     sort_key=inputters.str2sortkey[self.data_type],
        #     filter_pred=self._filter_pred,
        # )

        return self._translate(
            src,
            tgt=tgt,
            batch_size=batch_size,
            batch_type=batch_type,
            attn_debug=attn_debug,
            align_debug=align_debug,
            phrase_table=phrase_table,
            dynamic=True,
            transforms=transform,
        )

    def translate(
        self,
        src,
        src_feats={},
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
    ):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_feats: See :func`self.src_reader.read()`.
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        if self.tgt_prefix and tgt is None:
            raise ValueError("Prefix should be feed to tgt if -tgt_prefix.")

        # FIXME
        # src_data = {"reader": self.src_reader, "data": src, "features": src_feats}
        # tgt_data = {"reader": self.tgt_reader, "data": tgt, "features": {}}
        # _readers, _data = None, None  # inputters.Dataset.config([("src", src_data), ("tgt", tgt_data)])

        # data = inputters.Dataset(
        #     self.fields,
        #     readers=_readers,
        #     data=_data,
        #     sort_key=inputters.str2sortkey[self.data_type],
        #     filter_pred=self._filter_pred,
        # )

        return self._translate(
            src,
            tgt=tgt,
            batch_size=batch_size,
            batch_type=batch_type,
            attn_debug=False,
            align_debug=False,
            phrase_table="",
            transforms=None,
            dynamic=False,
        )

    def _translate(
        self,
        src,
        tgt=None,
        batch_size=None,
        batch_type="sents",
        attn_debug=False,
        align_debug=False,
        phrase_table="",
        transforms=None,
        dynamic=False,
    ):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_feats: See :func`self.src_reader.read()`.
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging
            align_debug (bool): enables the word alignment logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        self._log("src vocab: {}".format(self.vocabs['src']))
        self._log("transforms: {}".format(transforms))
        corpus = ParallelCorpus(
            src,
            tgt,
            self.vocabs['src'],
            self.vocabs['tgt'],
            transforms=transforms,  # I suppose you might want *some* transforms
            # batch_size=batch_size,
            # batch_type=batch_type,
            task=self.task,
            stride=1,
            offset=0,
        ).to(self._device)

        batches = build_dataloader(
            corpus,
            batch_size=batch_size,
            batch_type=batch_type,
            max_look_ahead_sentences=512,
            lookahead_minibatches=512,
            cycle=False,
        )

        # read_examples_from_files(None, None)

        # data_iter = inputters.OrderedIterator(
        #     dataset=data,
        #     device=self._device,
        #     batch_size=batch_size,
        #     batch_size_fn=max_tok_len if batch_type == "tokens" else None,
        #     train=False,
        #     sort=False,
        #     sort_within_batch=True,
        #     shuffle=False,
        # )
        # data_iter = None

        xlation_builder = TranslationBuilder(
            corpus,
            self.vocabs,
            self.n_best,
            self.replace_unk,
            has_tgt=tgt is not None,
            phrase_table=self.phrase_table,
        )

        # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for batch in batches:
            batch.to(corpus.device)
            batch_data = self.translate_batch(batch, corpus.vocabs['src'], attn_debug)
            translations = xlation_builder.from_batch(batch_data)

            for trans in translations:
                all_scores += [trans.pred_scores[: self.n_best]]
                pred_score_total += trans.pred_scores[0]
                pred_words_total += len(trans.pred_sents[0])
                if tgt is not None:
                    gold_score_total += trans.gold_score
                    gold_words_total += len(trans.gold_sent) + 1

                n_best_preds = [" ".join(pred) for pred in trans.pred_sents[: self.n_best]]
                if self.report_align:
                    align_pharaohs = [build_align_pharaoh(align) for align in trans.word_aligns[: self.n_best]]
                    n_best_preds_align = [" ".join(align) for align in align_pharaohs]
                    n_best_preds = [
                        pred + DefaultTokens.ALIGNMENT_SEPARATOR + align
                        for pred, align in zip(n_best_preds, n_best_preds_align)
                    ]

                if dynamic:
                    n_best_preds = [transforms.apply_reverse(x) for x in n_best_preds]
                all_predictions += [n_best_preds]
                self.out_file.write("\n".join(n_best_preds) + "\n")
                self.out_file.flush()

                if self.verbose:
                    sent_number = next(counter)
                    output = trans.log(sent_number)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if attn_debug:
                    preds = trans.pred_sents[0]
                    preds.append(DefaultTokens.EOS)
                    attns = trans.attns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(attns[0]))]
                    output = report_matrix(srcs, preds, attns)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

                if align_debug:
                    tgts = trans.pred_sents[0]
                    align = trans.word_aligns[0].tolist()
                    if self.data_type == "text":
                        srcs = trans.src_raw
                    else:
                        srcs = [str(item) for item in range(len(align[0]))]
                    output = report_matrix(srcs, tgts, align)
                    if self.logger:
                        self.logger.info(output)
                    else:
                        os.write(1, output.encode("utf-8"))

        end_time = time.time()

        if self.report_score:
            msg = self._report_score("PRED", pred_score_total, pred_words_total)
            self._log(msg)
            if tgt is not None:
                msg = self._report_score("GOLD", gold_score_total, gold_words_total)
                self._log(msg)

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (pred_words_total / total_time))

        if self.dump_beam:
            import json

            json.dump(
                self.translator.beam_accum,
                codecs.open(self.dump_beam, "w", "utf-8"),
            )
        return all_scores, all_predictions

    def _align_pad_prediction(self, predictions, bos, pad):
        """
        Padding predictions in batch and add BOS.

        Args:
            predictions (List[List[Tensor]]): `(batch, n_best,)`, for each src
                sequence contain n_best tgt predictions all of which ended with
                eos id.
            bos (int): bos index to be used.
            pad (int): pad index to be used.

        Return:
            batched_nbest_predict (torch.LongTensor): `(batch, n_best, tgt_l)`
        """
        dtype, device = predictions[0][0].dtype, predictions[0][0].device
        flatten_tgt = [best.tolist() for bests in predictions for best in bests]
        paded_tgt = torch.tensor(
            list(zip_longest(*flatten_tgt, fillvalue=pad)),
            dtype=dtype,
            device=device,
        ).T
        bos_tensor = torch.full([paded_tgt.size(0), 1], bos, dtype=dtype, device=device)
        full_tgt = torch.cat((bos_tensor, paded_tgt), dim=-1)
        batched_nbest_predict = full_tgt.view(len(predictions), -1, full_tgt.size(-1))  # (batch, n_best, tgt_l)
        return batched_nbest_predict

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            avg_score = score_total / words_total
            ppl = np.exp(-score_total.item() / words_total)
            msg = "%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name,
                avg_score,
                name,
                ppl,
            )
        return msg

    def _decode_and_generate(
        self,
        decoder_in,
        memory_bank,
        batch,
        src_vocabs,
        memory_lengths,
        src_map=None,
        step=None,
        batch_offset=None,
    ):
        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        log_probs = self.model.generator[f"generator_{self.task.tgt_lang}"](dec_out.squeeze(0))
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence

        return log_probs, attn

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        raise NotImplementedError

    def _score_target(
        self,
        batch,
        encoder_output,
        src_mask,
        src_vocabs,
        batch_size,
    ):
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        batch,
        batch_size,
        src_mask,
        src_vocabs,
        decode_strategy,
    ):
        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": gold_score,
        }

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        return results


class Translator(Inference):

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearch(
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    batch_size=batch.batch_size,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                    device=self._device,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearch(
                    self.beam_size,
                    batch_size=batch.batch_size,
                    pad=self._tgt_pad_idx,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                    device=self._device,
                )
            return self._translate_batch_with_strategy(batch, src_vocabs, decode_strategy)

    def _run_encoder(self, active_encoder, batch):
        src = rearrange(batch.src.tensor, 't b 1 -> b t')
        src_mask = rearrange(batch.src.mask, 't b -> b t')
        encoder_output = active_encoder(
            x=src,
            mask=src_mask,
            return_embeddings=True,
        )

        encoder_output, alphas = self.model.attention_bridge(encoder_output, src_mask)
        if self.model.attention_bridge.is_fixed_length:
            # turn off masking in the transformer decoder
            src_mask = None

        return encoder_output, src_mask

    def _translate_batch_with_strategy(self, batch, src_vocabs, decode_strategy):
        """Translate a batch of sentences step by step using cache.

        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.

        Returns:
            results (dict): The translation results.
        """
        # (0) Prep the components of the search.
        batch_size = batch.batch_size

        # (1) Activate the correct pluggable embeddings and modules
        metadata = self.task.get_serializable_metadata()
        active_encoder = self.model.encoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.encoder_adapter_ids,
        )
        active_decoder = self.model.decoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.decoder_adapter_ids,
        )
        active_encoder.to(self._device)
        active_decoder.to(self._device)

        # (2) Run the encoder on the src
        encoder_output, src_mask = self._run_encoder(active_encoder, batch)

        # (3) Decode and score the gold targets
        gold_score = self._gold_score(
            active_decoder,
            batch,
            encoder_output,
            src_mask,
            src_vocabs,
            batch_size,
        )

        # (4) prep decode_strategy
        # TODO: produce an optional target prefix
        # file contents or empty string -> transforms -> numericalize -> *left* pad (align right edge)
        # unfortunately AttentionLayers takes a seq_start_pos and constructs the mask, instead of taking a mask
        target_prefix = None
        seq_start_pos = None
        decode_strategy.initialize(
            target_prefix=target_prefix,
            encoder_output=encoder_output,
            src_mask=src_mask,
        )

        # (5) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.alive_seq

            logits_for_whole_sequence, new_cache = active_decoder(
                decoder_input,
                context=decode_strategy.encoder_output_tiled,
                context_mask=decode_strategy.src_mask_tiled,
                return_attn=False,
                return_embeddings=False,
                return_intermediates=True,
                cache=decode_strategy.cache,
                seq_start_pos=seq_start_pos,
            )
            # new_cache is a list of LayerIntermediates objects, one for each layer_stack

            if active_decoder.can_cache_kv:
                decode_strategy.set_cache(new_cache)

            # we only need the logits of the new prediction
            logits = logits_for_whole_sequence[:, -1]
            log_probs = torch.log_softmax(logits, dim=-1)

            decode_strategy.advance(log_probs)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src_mask,
            src_vocabs,
            decode_strategy,
        )

    def _score_target(
        self,
        active_decoder,
        batch,
        encoder_output,
        src_mask,
        src_vocabs,
        batch_size,
    ):
        tgt = batch.tgt.tensor
        decoder_input = tgt[:-1]

        logits, decoder_output = active_decoder(
            decoder_input,
            context=encoder_output,
            context_mask=src_mask,
            return_attn=False,
            return_logits_and_embeddings=True,
        )
        log_probs = torch.log_softmax(logits, dim=-1)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores
