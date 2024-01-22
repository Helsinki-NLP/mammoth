#!/usr/bin/env python
""" Translator Class and builder """
import codecs
import os
import time
import numpy as np
from itertools import count, zip_longest

import torch

import mammoth.model_builder
import mammoth.modules.decoder_ensemble
# from mammoth.inputters.text_dataset import InferenceDataIterator
from mammoth.translate.beam_search import BeamSearch, BeamSearchLM
from mammoth.translate.greedy_search import GreedySearch, GreedySearchLM
from mammoth.utils.misc import tile, set_random_seed, report_matrix
from mammoth.utils.alignment import extract_alignment, build_align_pharaoh
from mammoth.constants import ModelTask, DefaultTokens
from mammoth.inputters.dataset import ParallelCorpus
from mammoth.inputters.dataloader import build_dataloader


def build_translator(opts, task, report_score=True, logger=None, out_file=None):
    if out_file is None:
        outdir = os.path.dirname(opts.output)
        if outdir and not os.path.isdir(outdir):
            # FIXME use warnings instead
            logger.info('WARNING: output file directory does not exist... creating it.')
            os.makedirs(os.path.dirname(opts.output), exist_ok=True)
        out_file = codecs.open(opts.output, "w+", "utf-8")

    load_test_model = (
        mammoth.modules.decoder_ensemble.load_test_model if len(opts.models) > 3
        else mammoth.model_builder.load_test_multitask_model
    )
    if logger:
        logger.info(str(task))
    vocabs, model, model_opts = load_test_model(opts, task)

    scorer = mammoth.translate.GNMTGlobalScorer.from_opts(opts)

    if model_opts.model_task == ModelTask.LANGUAGE_MODEL:
        translator = GeneratorLM.from_opts(
            model,
            vocabs,
            opts,
            model_opts,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opts.report_align,
            report_score=report_score,
            logger=logger,
        )
    else:
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
        copy_attn (bool): Use copy attention.
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
        copy_attn=False,
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
        tgt_vocab = dict(self.vocabs)["tgt"]
        self._tgt_vocab = tgt_vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[DefaultTokens.EOS]
        self._tgt_pad_idx = self._tgt_vocab.stoi[DefaultTokens.PAD]
        self._tgt_bos_idx = self._tgt_vocab.stoi[DefaultTokens.BOS]
        self._tgt_unk_idx = self._tgt_vocab.stoi[DefaultTokens.UNK]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) if self._use_cuda else torch.device("cpu")

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

        self.copy_attn = copy_attn

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
        cls.validate_task(model_opts.model_task)

        return cls(
            model,
            vocabs,
            opts.src,
            tgt_file_path=opts.tgt,
            gpu=opts.gpu,
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
            copy_attn=model_opts.copy_attn,
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
        batch,
        memory_bank,
        src_lengths,
        src_vocabs,
        use_src_map,
        enc_states,
        batch_size,
        src,
    ):
        if batch.tgt is not None:
            gs = self._score_target(
                batch,
                memory_bank,
                src_lengths,
                src_vocabs,
                batch.src_map if use_src_map else None,
            )
            self.model.decoder.init_state(src, memory_bank, enc_states)
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
        ).to(self._dev)

        batches = build_dataloader(
            corpus,
            batch_size=batch_size,
            batch_type=batch_type,
            pool_size=512,
            n_buckets=512,
            cycle=False,
        )

        # read_examples_from_files(None, None)

        # data_iter = inputters.OrderedIterator(
        #     dataset=data,
        #     device=self._dev,
        #     batch_size=batch_size,
        #     batch_size_fn=max_tok_len if batch_type == "tokens" else None,
        #     train=False,
        #     sort=False,
        #     sort_within_batch=True,
        #     shuffle=False,
        # )
        # data_iter = None

        xlation_builder = mammoth.translate.TranslationBuilder(
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
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx)

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

    def _score_target(self, batch, memory_bank, src_lengths, src_vocabs, src_map):
        raise NotImplementedError

    def report_results(
        self,
        gold_score,
        batch,
        batch_size,
        src,
        src_lengths,
        src_vocabs,
        use_src_map,
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
        if self.report_align:
            results["alignment"] = self._align_forward(batch, decode_strategy.predictions)
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        return results


class Translator(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.SEQ2SEQ:
            raise ValueError(f"Translator does not support task {task}. Tasks supported: {ModelTask.SEQ2SEQ}")

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        # (0) add BOS and padding to tgt prediction
        batch_tgt_idxs = self._align_pad_prediction(predictions, bos=self._tgt_bos_idx, pad=self._tgt_pad_idx)
        tgt_mask = (
            batch_tgt_idxs.eq(self._tgt_pad_idx)
            | batch_tgt_idxs.eq(self._tgt_eos_idx)
            | batch_tgt_idxs.eq(self._tgt_bos_idx)
        )

        n_best = batch_tgt_idxs.size(1)
        # (1) Encoder forward.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        # (2) Repeat src objects `n_best` times.
        # We use batch_size x n_best, get ``(src_len, batch * n_best, nfeat)``
        src = tile(src, n_best, dim=1)
        enc_states = tile(enc_states, n_best, dim=1)
        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, n_best, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, n_best, dim=1)
        src_lengths = tile(src_lengths, n_best)  # ``(batch * n_best,)``

        # (3) Init decoder with n_best src,
        self.model.decoder.init_state(src, memory_bank, enc_states)
        # reshape tgt to ``(len, batch * n_best, nfeat)``
        tgt = batch_tgt_idxs.view(-1, batch_tgt_idxs.size(-1)).T.unsqueeze(-1)
        dec_in = tgt[:-1]  # exclude last target from inputs
        _, attns = self.model.decoder(dec_in, memory_bank, memory_lengths=src_lengths, with_align=True)

        alignment_attn = attns["align"]  # ``(B, tgt_len-1, src_len)``
        # masked_select
        align_tgt_mask = tgt_mask.view(-1, tgt_mask.size(-1))
        prediction_mask = align_tgt_mask[:, 1:]  # exclude bos to match pred
        # get aligned src id for each prediction's valid tgt tokens
        alignement = extract_alignment(alignment_attn, prediction_mask, src_lengths, n_best)
        return alignement

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
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
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
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(batch, src_vocabs, decode_strategy)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

        enc_states, memory_bank, src_lengths, mask = self.model.encoder(
            src, src_lengths
        )

        memory_bank, alphas = self.model.attention_bridge(memory_bank, mask)

        if src_lengths is None or self.model.attention_bridge.is_fixed_length:
            assert not isinstance(memory_bank, tuple), "Ensemble decoding only supported for text data"
            src_lengths = torch.Tensor(batch.batch_size).type_as(memory_bank).long().fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

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
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (0.5) Activate adapters
        metadata = self.task.get_serializable_metadata()
        self.model.encoder.activate(metadata)
        self.model.decoder.activate(metadata)

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        gold_score = self._gold_score(
            batch,
            memory_bank,
            src_lengths,
            src_vocabs,
            use_src_map,
            enc_states,
            batch_size,
            src,
        )

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        target_prefix = batch.tgt if self.tgt_prefix else None
        (
            fn_map_state,
            memory_bank,
            memory_lengths,
            src_map,
        ) = decode_strategy.initialize(memory_bank, src_lengths, src_map, target_prefix=target_prefix)
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=decode_strategy.batch_offset,
            )

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices) for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )

    def _score_target(self, batch, memory_bank, src_lengths, src_vocabs, src_map):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in,
            memory_bank,
            batch,
            src_vocabs,
            memory_lengths=src_lengths,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores


class GeneratorLM(Inference):
    @classmethod
    def validate_task(cls, task):
        if task != ModelTask.LANGUAGE_MODEL:
            raise ValueError(
                f"GeneratorLM does not support task {task}. Tasks supported: {ModelTask.LANGUAGE_MODEL}"
            )

    def _align_forward(self, batch, predictions):
        """
        For a batch of input and its prediction, return a list of batch predict
        alignment src indice Tensor in size ``(batch, n_best,)``.
        """
        raise NotImplementedError

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
        if batch_size != 1:
            warning_msg = (
                "GeneratorLM does not support batch_size != 1"
                " nicely. You can remove this limitation here."
                " With batch_size > 1 the end of each input is"
                " repeated until the input is finished. Then"
                " generation will start."
            )
            if self.logger:
                self.logger.info(warning_msg)
            else:
                os.write(1, warning_msg.encode("utf-8"))

        return super(GeneratorLM, self).translate(
            src,
            src_feats,
            tgt,
            batch_size=1,
            batch_type=batch_type,
            attn_debug=attn_debug,
            align_debug=align_debug,
            phrase_table=phrase_table,
        )

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if self.sample_from_topk != 0 or self.sample_from_topp != 0:
                decode_strategy = GreedySearchLM(
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
                    return_attention=attn_debug or self.replace_unk,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=self.beam_size,
                    ban_unk_token=self.ban_unk_token,
                )
            else:
                # TODO: support these blacklisted features
                assert not self.dump_beam
                decode_strategy = BeamSearchLM(
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
                    return_attention=attn_debug or self.replace_unk,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                )
            return self._translate_batch_with_strategy(batch, src_vocabs, decode_strategy)

    @classmethod
    def split_src_to_prevent_padding(cls, src, src_lengths):
        min_len_batch = torch.min(src_lengths).item()
        target_prefix = None
        if min_len_batch > 0 and min_len_batch < src.size(0):
            target_prefix = src[min_len_batch:]
            src = src[:min_len_batch]
            src_lengths[:] = min_len_batch
        return src, src_lengths, target_prefix

    def tile_to_beam_size_after_initial_step(self, fn_map_state, log_probs):
        if fn_map_state is not None:
            log_probs = fn_map_state(log_probs, dim=1)
            self.model.decoder.map_state(fn_map_state)
            log_probs = log_probs[-1]
        return log_probs

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
        use_src_map = self.copy_attn
        parallel_paths = decode_strategy.parallel_paths  # beam_size
        batch_size = batch.batch_size

        # (1) split src into src and target_prefix to avoid padding.
        src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

        src, src_lengths, target_prefix = self.split_src_to_prevent_padding(src, src_lengths)

        # (2) init decoder
        self.model.decoder.init_state(src, None, None)
        gold_score = self._gold_score(
            batch,
            None,
            src_lengths,
            src_vocabs,
            use_src_map,
            None,
            batch_size,
            src,
        )

        # (3) prep decode_strategy. Possibly repeat src objects.
        src_map = batch.src_map if use_src_map else None
        (fn_map_state, src, memory_lengths, src_map,) = decode_strategy.initialize(
            src,
            src_lengths,
            src_map,
            target_prefix=target_prefix,
        )

        # (4) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = src if step == 0 else decode_strategy.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                None,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths.clone(),
                src_map=src_map,
                step=step if step == 0 else step + src_lengths[0].item(),
                batch_offset=decode_strategy.batch_offset,
            )

            if step == 0:
                log_probs = self.tile_to_beam_size_after_initial_step(fn_map_state, log_probs)

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            memory_lengths += 1
            if any_finished:
                # Reorder states.
                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                # select indexes in model state/cache
                self.model.decoder.map_state(lambda state, dim: state.index_select(dim, select_indices))

        return self.report_results(
            gold_score,
            batch,
            batch_size,
            src,
            src_lengths,
            src_vocabs,
            use_src_map,
            decode_strategy,
        )

    def _score_target(self, batch, memory_bank, src_lengths, src_vocabs, src_map):
        tgt = batch.tgt
        src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

        log_probs, attn = self._decode_and_generate(
            src,
            None,
            batch,
            src_vocabs,
            memory_lengths=src_lengths,
            src_map=src_map,
        )

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold_scores = log_probs.gather(2, tgt)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores
