import unittest
from mammoth.translate.beam_search import BeamSearch, GNMTGlobalScorer

from copy import deepcopy

import torch


class GlobalScorerStub(object):
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.0
        self.cov_penalty = lambda cov, beta: torch.zeros((1, cov.shape[-2]), device=cov.device, dtype=torch.float)
        self.has_cov_pen = False
        self.has_len_pen = False

    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores


class TestBeamSearch(unittest.TestCase):
    BLOCKED_SCORE = -10e20

    def test_advance_with_all_repeats_gets_blocked(self):
        # all beams repeat (beam >= 1 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        src_len = 71
        device_init = torch.zeros(1, 1)
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_size=beam_sz,
                batch_size=batch_sz,
                pad=0,
                bos=1,
                eos=2,
                unk=3,
                n_best=2,
                global_scorer=GlobalScorerStub(),
                min_length=0,
                max_length=30,
                block_ngram_repeat=ngram_repeat,
                exclusion_tokens=set(),
                stepwise_penalty=False,
                ratio=0.0,
                ban_unk_token=False,
                device=device_init.device,
            )
            beam.initialize(
                target_prefix=torch.randint(0, 30, (batch_sz,)),
                encoder_output=torch.randn(batch_sz, src_len, 73),
                src_mask=torch.randint(0, 1, (batch_sz, src_len))
            )
            for i in range(ngram_repeat + 4):
                # predict repeat_idx over and over again
                word_probs = torch.full((batch_sz * beam_sz, n_words), -float('inf'))
                word_probs[:, repeat_idx] = 0

                # TODO: test that LayerIntermediates is correctly mangled
                # attns = torch.randn(1, batch_sz * beam_sz, 53)
                # beam.set_cache(attns)
                beam.advance(word_probs)

                if i < ngram_repeat:
                    # before repeat, scores are either 0 or -inf
                    expected_scores = torch.tensor([0] + [-float('inf')] * (beam_sz - 1)).repeat(batch_sz, 1)
                    self.assertTrue(beam.topk_log_probs.equal(expected_scores))
                elif i % ngram_repeat == 0:
                    # on repeat, `repeat_idx` score is BLOCKED_SCORE
                    # (but it's still the best score, thus we have
                    # [BLOCKED_SCORE, -inf, -inf, -inf, -inf]
                    expected_scores = torch.tensor([self.BLOCKED_SCORE] + [-float('inf')] * (beam_sz - 1)).repeat(
                        batch_sz, 1
                    )
                    self.assertTrue(beam.topk_log_probs.equal(expected_scores))
                else:
                    # repetitions keeps maximizing score
                    # index 0 has been blocked, so repeating=>+0.0 score
                    # other indexes are -inf so repeating=>BLOCKED_SCORE
                    # which is higher
                    expected_scores = torch.tensor([self.BLOCKED_SCORE] + [-float('inf')] * (beam_sz - 1)).repeat(
                        batch_sz, 1
                    )
                    self.assertTrue(beam.topk_log_probs.equal(expected_scores))

    def test_advance_with_some_repeats_gets_blocked(self):
        # beam 0 and beam >=2 will repeat (beam >= 2 repeat dummy scores)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47
        ngram_repeat = 3
        no_repeat_score = -2.3
        repeat_score = -0.1
        src_len = 79
        device_init = torch.zeros(1, 1)
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_size=beam_sz,
                batch_size=batch_sz,
                pad=0,
                bos=1,
                eos=2,
                unk=3,
                n_best=2,
                global_scorer=GlobalScorerStub(),
                min_length=0,
                max_length=30,
                block_ngram_repeat=ngram_repeat,
                exclusion_tokens=set(),
                stepwise_penalty=False,
                ratio=0.0,
                ban_unk_token=False,
                device=device_init.device,
            )
            beam.initialize(
                target_prefix=torch.randint(0, 30, (batch_sz,)),
                encoder_output=torch.randn(batch_sz, src_len, 73),
                src_mask=torch.randint(0, 1, (batch_sz, src_len))
            )
            for i in range(ngram_repeat + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full((batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    # on initial round, only predicted scores for beam 0
                    # matter. Make two predictions. Top one will be repeated
                    # in beam zero, second one will live on in beam 1.
                    word_probs[0::beam_sz, repeat_idx] = repeat_score
                    word_probs[0::beam_sz, repeat_idx + i + 1] = no_repeat_score
                else:
                    # predict the same thing in beam 0
                    word_probs[0::beam_sz, repeat_idx] = 0
                    # continue pushing around what beam 1 predicts
                    word_probs[1::beam_sz, repeat_idx + i + 1] = 0
                # TODO: test that LayerIntermediates is correctly mangled
                # attns = torch.randn(1, batch_sz * beam_sz, 53)
                # beam.set_cache(attns)
                beam.advance(word_probs)
                if i < ngram_repeat:
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(self.BLOCKED_SCORE).any())
                    self.assertFalse(beam.topk_log_probs[:, 1].eq(self.BLOCKED_SCORE).any())
                elif i == ngram_repeat:
                    # now beam 0 dies (along with the others), beam 1 -> beam 0
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(self.BLOCKED_SCORE).any())

                    expected = torch.full([batch_sz, beam_sz], float("-inf"))
                    expected[:, 0] = no_repeat_score
                    expected[:, 1] = self.BLOCKED_SCORE
                    self.assertTrue(beam.topk_log_probs.equal(expected))
                else:
                    # now beam 0 dies (along with the others), beam 1 -> beam 0
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(self.BLOCKED_SCORE).any())

                    expected = torch.full([batch_sz, beam_sz], float("-inf"))
                    expected[:, 0] = no_repeat_score
                    expected[:, 1:3] = self.BLOCKED_SCORE
                    expected[:, 3:] = float("-inf")
                    self.assertTrue(beam.topk_log_probs.equal(expected))

    def test_repeating_excluded_index_does_not_die(self):
        # beam 0 and beam >= 2 will repeat (beam 2 repeats excluded idx)
        beam_sz = 5
        n_words = 100
        repeat_idx = 47  # will be repeated and should be blocked
        repeat_idx_ignored = 7  # will be repeated and should not be blocked
        ngram_repeat = 3
        src_len = 71
        device_init = torch.zeros(1, 1)
        for batch_sz in [1, 3]:
            beam = BeamSearch(
                beam_size=beam_sz,
                batch_size=batch_sz,
                pad=0,
                bos=1,
                eos=2,
                unk=3,
                n_best=2,
                global_scorer=GlobalScorerStub(),
                min_length=0,
                max_length=30,
                block_ngram_repeat=ngram_repeat,
                exclusion_tokens={repeat_idx_ignored},
                stepwise_penalty=False,
                ratio=0.0,
                ban_unk_token=False,
                device=device_init.device,
            )
            beam.initialize(
                target_prefix=torch.randint(0, 30, (batch_sz,)),
                encoder_output=torch.randn(batch_sz, src_len, 73),
                src_mask=torch.randint(0, 1, (batch_sz, src_len))
            )
            for i in range(ngram_repeat + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full((batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    word_probs[0::beam_sz, repeat_idx] = -0.1
                    word_probs[0::beam_sz, repeat_idx + i + 1] = -2.3
                    word_probs[0::beam_sz, repeat_idx_ignored] = -5.0
                else:
                    # predict the same thing in beam 0
                    word_probs[0::beam_sz, repeat_idx] = 0
                    # continue pushing around what beam 1 predicts
                    word_probs[1::beam_sz, repeat_idx + i + 1] = 0
                    # predict the allowed-repeat again in beam 2
                    word_probs[2::beam_sz, repeat_idx_ignored] = 0
                # TODO: test that LayerIntermediates is correctly mangled
                # attns = torch.randn(1, batch_sz * beam_sz, 53)
                # beam.set_cache(attns)
                beam.advance(word_probs)
                if i < ngram_repeat:
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(self.BLOCKED_SCORE).any())
                    self.assertFalse(beam.topk_log_probs[:, 1].eq(self.BLOCKED_SCORE).any())
                    self.assertFalse(beam.topk_log_probs[:, 2].eq(self.BLOCKED_SCORE).any())
                else:
                    # now beam 0 dies, beam 1 -> beam 0, beam 2 -> beam 1
                    # and the rest die
                    self.assertFalse(beam.topk_log_probs[:, 0].eq(self.BLOCKED_SCORE).any())
                    # since all preds after i=0 are 0, we can check
                    # that the beam is the correct idx by checking that
                    # the curr score is the initial score
                    self.assertTrue(beam.topk_log_probs[:, 0].eq(-2.3).all())
                    self.assertFalse(beam.topk_log_probs[:, 1].eq(self.BLOCKED_SCORE).all())
                    self.assertTrue(beam.topk_log_probs[:, 1].eq(-5.0).all())

                    self.assertTrue(beam.topk_log_probs[:, 2].eq(self.BLOCKED_SCORE).all())

    def test_doesnt_predict_eos_if_shorter_than_min_len(self):
        # beam 0 will always predict EOS. The other beams will predict
        # non-eos scores.
        for batch_sz in [1, 3]:
            beam_sz = 5
            n_words = 100
            _non_eos_idxs = [47, 51, 13, 88, 99]
            valid_score_dist = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
            min_length = 5
            eos_idx = 2
            src_len = 71
            device_init = torch.zeros(1, 1)
            beam = BeamSearch(
                beam_size=beam_sz,
                batch_size=batch_sz,
                pad=0,
                bos=1,
                eos=2,
                unk=3,
                n_best=2,
                global_scorer=GlobalScorerStub(),
                min_length=min_length,
                max_length=30,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=False,
                ratio=0.0,
                ban_unk_token=False,
                device=device_init.device,
            )
            beam.initialize(
                encoder_output=torch.randn(batch_sz, src_len, 73),
                src_mask=torch.randint(0, 1, (batch_sz, src_len))
            )
            for i in range(min_length + 4):
                # non-interesting beams are going to get dummy values
                word_probs = torch.full((batch_sz * beam_sz, n_words), -float('inf'))
                if i == 0:
                    # "best" prediction is eos - that should be blocked
                    word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                    # include at least beam_sz predictions OTHER than EOS
                    # that are greater than -1e20
                    for j, score in zip(_non_eos_idxs, valid_score_dist[1:]):
                        word_probs[0::beam_sz, j] = score
                else:
                    # predict eos in beam 0
                    word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                    # provide beam_sz other good predictions
                    for k, (j, score) in enumerate(zip(_non_eos_idxs, valid_score_dist[1:])):
                        beam_idx = min(beam_sz - 1, k)
                        word_probs[beam_idx::beam_sz, j] = score

                # TODO: test that LayerIntermediates is correctly mangled
                # attns = torch.randn(1, batch_sz * beam_sz, 53)
                # beam.set_cache(attns)
                beam.advance(word_probs)
                if i < min_length:
                    expected_score_dist = (i + 1) * valid_score_dist[1:].unsqueeze(0)
                    # Note that when batch_sz is > 1, expected is broadcast across the batch
                    self.assertTrue(beam.topk_log_probs.allclose(expected_score_dist))
                    # self.assertTrue(beam.cache.shape == torch.Size([1, batch_sz * beam_sz, 53]))
                elif i == min_length:
                    # now the top beam has ended and no others have
                    self.assertTrue(beam.is_finished[:, 0].eq(1).all())
                    self.assertTrue(beam.is_finished[:, 1:].eq(0).all())
                    # self.assertTrue(beam.cache.shape == torch.Size([1, batch_sz * (beam_sz - 1), 53]))
                else:  # i > min_length
                    # not of interest, but want to make sure it keeps running
                    # since only beam 0 terminates and n_best = 2
                    pass

    def test_beam_is_done_when_n_best_beams_eos_using_min_length(self):
        # this is also a test that when block_ngram_repeat=0,
        # repeating is acceptable
        beam_sz = 5
        batch_sz = 3
        n_words = 100
        _non_eos_idxs = [47, 51, 13, 88, 99]
        valid_score_dist = torch.log_softmax(torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), dim=0)
        min_length = 5
        eos_idx = 2
        src_len = 71
        device_init = torch.zeros(1, 1)
        beam = BeamSearch(
            beam_size=beam_sz,
            batch_size=batch_sz,
            pad=0,
            bos=1,
            eos=2,
            unk=3,
            n_best=2,
            global_scorer=GlobalScorerStub(),
            min_length=min_length,
            max_length=30,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            stepwise_penalty=False,
            ratio=0.0,
            ban_unk_token=False,
            device=device_init.device,
        )
        beam.initialize(
            target_prefix=torch.randint(0, 30, (batch_sz,)),
            encoder_output=torch.randn(batch_sz, src_len, 73),
            src_mask=torch.randint(0, 1, (batch_sz, src_len))
        )
        for i in range(min_length + 4):
            # non-interesting beams are going to get dummy values
            word_probs = torch.full((batch_sz * beam_sz, n_words), -float('inf'))
            if i == 0:
                # "best" prediction is eos - that should be blocked
                word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                # include at least beam_sz predictions OTHER than EOS
                # that are greater than -1e20
                for j, score in zip(_non_eos_idxs, valid_score_dist[1:]):
                    word_probs[0::beam_sz, j] = score
            elif i <= min_length:
                # predict eos in beam 1
                word_probs[1::beam_sz, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz - 1, k)
                    word_probs[beam_idx::beam_sz, j] = score
            else:
                word_probs[0::beam_sz, eos_idx] = valid_score_dist[0]
                word_probs[1::beam_sz, eos_idx] = valid_score_dist[0]
                # provide beam_sz other good predictions in other beams
                for k, (j, score) in enumerate(zip(_non_eos_idxs, valid_score_dist[1:])):
                    beam_idx = min(beam_sz - 1, k)
                    word_probs[beam_idx::beam_sz, j] = score

            # TODO: test that LayerIntermediates is correctly mangled
            # attns = torch.randn(1, batch_sz * beam_sz, 53)
            # beam.set_cache(attns)
            beam.advance(word_probs)
            if i < min_length:
                self.assertFalse(beam.done)
            elif i == min_length:
                # beam 1 dies on min_length
                self.assertTrue(beam.is_finished[:, 1].all())
                beam.update_finished()
                self.assertFalse(beam.done)
            else:  # i > min_length
                # beam 0 dies on the step after beam 1 dies
                self.assertTrue(beam.is_finished[:, 0].all())
                beam.update_finished()
                self.assertTrue(beam.done)


class TestBeamSearchAgainstReferenceCase(unittest.TestCase):
    # this is just test_beam.TestBeamAgainstReferenceCase repeated
    # in each batch.
    BEAM_SZ = 5
    EOS_IDX = 2  # don't change this - all the scores would need updated
    N_WORDS = 8  # also don't change for same reason
    N_BEST = 3
    DEAD_SCORE = -1e20
    BATCH_SZ = 3
    INP_SEQ_LEN = 53

    def random_attn(self):
        return torch.randn(1, self.BATCH_SZ * self.BEAM_SZ, self.INP_SEQ_LEN)

    def init_step(self, beam, expected_len_pen):
        # init_preds: [4, 3, 5, 6, 7] - no EOS's
        init_scores = torch.log_softmax(torch.tensor([[0, 0, 0, 4, 5, 3, 2, 1]], dtype=torch.float), dim=1)
        init_scores = deepcopy(init_scores.repeat(self.BATCH_SZ * self.BEAM_SZ, 1))
        new_scores = init_scores + beam.topk_log_probs.view(-1).unsqueeze(1)
        expected_beam_scores, expected_preds_0 = new_scores.view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS).topk(
            self.BEAM_SZ, dim=-1
        )
        beam.advance(deepcopy(init_scores))
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_ids.equal(expected_preds_0))
        self.assertFalse(beam.is_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def first_step(self, beam, expected_beam_scores, expected_len_pen):
        # no EOS's yet
        assert beam.is_finished.sum() == 0
        scores_1 = torch.log_softmax(
            torch.tensor(
                [
                    [0, 0, 0, 0.3, 0, 0.51, 0.2, 0],
                    [0, 0, 1.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.49, 0.48, 0, 0],
                    [0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2],
                    [0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2],
                ]
            ),
            dim=1,
        )
        scores_1 = scores_1.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_1))

        new_scores = scores_1 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores.view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS).topk(
            self.BEAM_SZ, -1
        )
        expected_bptr_1 = unreduced_preds // self.N_WORDS
        # [5, 3, 2, 6, 0], so beam 2 predicts EOS!
        expected_preds_1 = unreduced_preds - expected_bptr_1 * self.N_WORDS
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_1))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_1))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        self.assertTrue(beam.is_finished[:, 2].all())  # beam 2 finished
        beam.update_finished()
        self.assertFalse(beam.top_beam_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def second_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 2 finished on last step
        scores_2 = torch.log_softmax(
            torch.tensor(
                [
                    [0, 0, 0, 0.3, 0, 0.51, 0.2, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 5000, 0.48, 0, 0],  # beam 2 shouldn't continue
                    [0, 0, 50, 0.2, 0.2, 0.2, 0.2, 0.2],  # beam 3 -> beam 0 should die
                    [0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2],
                ]
            ),
            dim=1,
        )
        scores_2 = scores_2.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_2))

        # ended beam 2 shouldn't continue
        expected_beam_scores[:, 2::self.BEAM_SZ] = self.DEAD_SCORE
        new_scores = scores_2 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores.view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS).topk(
            self.BEAM_SZ, -1
        )
        expected_bptr_2 = unreduced_preds // self.N_WORDS
        # [2, 5, 3, 6, 0] repeat self.BATCH_SZ, so beam 0 predicts EOS!
        expected_preds_2 = unreduced_preds - expected_bptr_2 * self.N_WORDS
        # [-2.4879, -3.8910, -4.1010, -4.2010, -4.4010] repeat self.BATCH_SZ
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_2))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_2))
        # another beam is finished in all batches
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        # new beam 0 finished
        self.assertTrue(beam.is_finished[:, 0].all())
        # new beam 0 is old beam 3
        self.assertTrue(expected_bptr_2[:, 0].eq(3).all())
        beam.update_finished()
        self.assertTrue(beam.top_beam_finished.all())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def third_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 0 finished on last step
        scores_3 = torch.log_softmax(
            torch.tensor(
                [
                    [0, 0, 5000, 0, 5000, 0.51, 0.2, 0],  # beam 0 shouldn't cont
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 5000, 0, 0],
                    [0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2],
                    [0, 0, 50, 0, 0.2, 0.2, 0.2, 0.2],
                ]  # beam 4 -> beam 1 should die
            ),
            dim=1,
        )
        scores_3 = scores_3.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_3))

        expected_beam_scores[:, 0::self.BEAM_SZ] = self.DEAD_SCORE
        new_scores = scores_3 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores.view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS).topk(
            self.BEAM_SZ, -1
        )
        expected_bptr_3 = unreduced_preds // self.N_WORDS
        # [5, 2, 6, 1, 0] repeat self.BATCH_SZ, so beam 1 predicts EOS!
        expected_preds_3 = unreduced_preds - expected_bptr_3 * self.N_WORDS
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_3))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_3))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        # new beam 1 finished
        self.assertTrue(beam.is_finished[:, 1].all())
        # new beam 1 is old beam 4
        self.assertTrue(expected_bptr_3[:, 1].eq(4).all())
        beam.update_finished()
        self.assertTrue(beam.top_beam_finished.all())
        self.assertTrue(beam.done)
        return expected_beam_scores

    def test_beam_advance_against_known_reference(self):
        src_len = 71
        device_init = torch.zeros(1, 1)
        beam = BeamSearch(
            beam_size=self.BEAM_SZ,
            batch_size=self.BATCH_SZ,
            pad=0,
            bos=1,
            eos=2,
            unk=3,
            n_best=self.N_BEST,
            global_scorer=GlobalScorerStub(),
            min_length=0,
            max_length=30,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            stepwise_penalty=False,
            ratio=0.0,
            ban_unk_token=False,
            device=device_init.device,
        )
        beam.initialize(
            target_prefix=torch.randint(0, 30, (self.BATCH_SZ,)),
            encoder_output=torch.randn(self.BATCH_SZ, src_len, 73),
            src_mask=torch.randint(0, 1, (self.BATCH_SZ, src_len))
        )
        expected_beam_scores = self.init_step(beam, 1)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 1)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 1)
        self.third_step(beam, expected_beam_scores, 1)


class TestBeamWithLengthPenalty(TestBeamSearchAgainstReferenceCase):
    # this could be considered an integration test because it tests
    # interactions between the GNMT scorer and the beam

    def test_beam_advance_against_known_reference(self):
        scorer = GNMTGlobalScorer(0.7, 0.0, "avg", "none")
        src_len = 71
        device_init = torch.zeros(1, 1)
        beam = BeamSearch(
            beam_size=self.BEAM_SZ,
            batch_size=self.BATCH_SZ,
            pad=0,
            bos=1,
            eos=2,
            unk=3,
            n_best=self.N_BEST,
            global_scorer=scorer,
            min_length=0,
            max_length=30,
            block_ngram_repeat=0,
            exclusion_tokens=set(),
            stepwise_penalty=False,
            ratio=0.0,
            ban_unk_token=False,
            device=device_init.device,
        )
        beam.initialize(
            target_prefix=torch.randint(0, 30, (self.BATCH_SZ,)),
            encoder_output=torch.randn(self.BATCH_SZ, src_len, 73),
            src_mask=torch.randint(0, 1, (self.BATCH_SZ, src_len))
        )
        expected_beam_scores = self.init_step(beam, 1.0)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 3)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 4)
        self.third_step(beam, expected_beam_scores, 5)
