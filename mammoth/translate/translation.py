""" Translation main class """
from mammoth.constants import DefaultTokens
from mammoth.utils.alignment import build_align_pharaoh


# FIXME
class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Args:
       data (mammoth.inputters.ParallelCorpus): Data.
       vocabs (dict[str, mammoth.inputters.Vocab]): data vocabs
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, vocabs, n_best=1, replace_unk=False, has_tgt=False, phrase_table=""):
        # FIXME: clean these up
        assert not replace_unk
        assert phrase_table == ""
        self.data = data
        self.vocabs = vocabs
        self._has_text_src = True  # isinstance(dict(self.fields)["src"], None)
        self.n_best = n_best
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred):
        vocab = self.vocabs['tgt']
        tokens = []

        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok.item()])
            else:
                tokens.append(src_vocab.itos[tok.item() - len(vocab)])
            if tokens[-1] == DefaultTokens.EOS:
                tokens = tokens[:-1]
                break
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert len(translation_batch["gold_score"]) == len(translation_batch["predictions"])
        batch_size = batch.batch_size

        preds, pred_score, gold_score = list(
            zip(
                *sorted(
                    zip(
                        translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["gold_score"],
                    ),
                    key=lambda x: x[-1],
                )
            )
        )

        # Sorting
        # inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0]  # .index_select(1, perm)
        else:
            src = None
        tgt = batch.tgt[:, :, 0] if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            # if self._has_text_src:
            #     src_vocab = self.data.vocabs['src']
            #     src_raw = self.data.examples[inds[b]].src[0]
            # else:
            src_vocab = None
            src_raw = None
            pred_sents = [
                self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab,
                    src_raw,
                    preds[b][n],
                )
                for n in range(self.n_best)
            ]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab,
                    src_raw,
                    tgt[1:, b] if tgt is not None else None,
                    None,
                )

            translation = Translation(
                src[:, b] if src is not None else None,
                src_raw,
                pred_sents,
                pred_score[b],
                gold_sent,
                gold_score[b],
            )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
    """

    __slots__ = ["src", "src_raw", "pred_sents", "pred_scores", "gold_sent", "gold_score"]

    def __init__(self, src, src_raw, pred_sents, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.word_aligns is not None:
            pred_align = self.word_aligns[0]
            pred_align_pharaoh = build_align_pharaoh(pred_align)
            pred_align_sent = ' '.join(pred_align_pharaoh)
            msg.append("ALIGN: {}\n".format(pred_align_sent))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)
