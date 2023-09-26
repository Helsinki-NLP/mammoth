import ast
import logging
import os
import shutil
from typing import List
from unittest import TestCase

import timeout_decorator

import mammoth
from mammoth.bin.train import train
from mammoth.utils.parse import ArgumentParser

import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


class TestTraining(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = ArgumentParser(description="train.py")
        mammoth.opts.train_opts(cls.parser)
        # clear output folders
        for folder in ["models", "tensorboard"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)

    def tearDown(self) -> None:
        # clear output folders
        for folder in ["models", "tensorboard"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        for child_process in mp.active_children():
            child_process.kill()

    @staticmethod
    def _get_model_components(opts) -> List[str]:
        # N.B: These components are only valid for very vanilla language-specific xcoder with fully shared AB models
        components_enc = [f"encoder_0_{src_lang}" for src_lang in ast.literal_eval(opts.src_vocab).keys()]
        components_dec = [f"encoder_0_{tgt_lang}" for tgt_lang in ast.literal_eval(opts.tgt_vocab).keys()]
        components_gen = [f"generator_{tgt_lang}" for tgt_lang in ast.literal_eval(opts.tgt_vocab).keys()]
        components_src_emb = [f"src_embeddings_{src_lang}" for src_lang in ast.literal_eval(opts.src_vocab).keys()]
        components_tgt_emb = [f"tgt_embeddings_{tgt_lang}" for tgt_lang in ast.literal_eval(opts.tgt_vocab).keys()]
        return [
            "frame",
            "attention_bridge",
            *components_enc,
            *components_dec,
            *components_gen,
            *components_src_emb,
            *components_tgt_emb,
        ]

    @timeout_decorator.timeout(60)
    def test_training_1gpu_4pairs(self):
        out_model_prefix = "wmt_1gpu_4pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_1gpu_4pairs_ab_lin(self):
        out_model_prefix = "wmt_1gpu_4pairs_lin"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "lin",
                "-hidden_ab_size",
                "512",
                "-ab_fixed_length",
                "50",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_1gpu_4pairs_ab_ff(self):
        out_model_prefix = "wmt_1gpu_4pairs_ff"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "feedforward",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_1gpu_4pairs_ab_tf(self):
        out_model_prefix = "wmt_1gpu_4pairs_tf"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "transformer",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_1gpu_4pairs_ab_simple(self):
        out_model_prefix = "wmt_1gpu_4pairs_simple"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "simple",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_1gpu_4pairs_ab_perceiver(self):
        out_model_prefix = "wmt_1gpu_4pairs_perceiver"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "perceiver",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs(self):
        out_model_prefix = "wmt_2gpus_4pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs_ab_lin(self):
        out_model_prefix = "wmt_2gpus_4pairs_lin"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "lin",
                "-hidden_ab_size",
                "512",
                "-ab_fixed_length",
                "50",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs_ab_ff(self):
        out_model_prefix = "wmt_2gpus_4pairs_ff"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "feedforward",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs_ab_tf(self):
        out_model_prefix = "wmt_2gpus_4pairs_tf"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "transformer",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs_ab_simple(self):
        out_model_prefix = "wmt_2gpus_4pairs_simple"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "simple",
                "-hidden_ab_size",
                "512",
                "-ab_fixed_length",
                "50",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs_ab_perceiver(self):
        out_model_prefix = "wmt_2gpus_4pairs_perceiver"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-ab_layers",
                "perceiver",
                "-hidden_ab_size",
                "512",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_2gpus_4pairs_crossed(self):
        out_model_prefix = "wmt_2gpus_4pairs_crossed"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:1",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(60)
    def test_training_4gpus_4pairs(self):
        out_model_prefix = "wmt_4gpus_4pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "4",
                "-gpu_ranks",
                "0",
                "1",
                "2",
                "3",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:2",
                "0:3",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(120)
    def test_training_3gpus_12pairs(self):
        out_model_prefix = "wmt_3gpus_12pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_12pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "3",
                "-gpu_ranks",
                "0",
                "1",
                "2",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(120)
    def test_training_3gpus_21pairs(self):
        out_model_prefix = "wmt_3gpus_21pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_21pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "3",
                "-gpu_ranks",
                "0",
                "1",
                "2",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(120)
    def test_training_4gpus_12pairs(self):
        out_model_prefix = "wmt_4gpus_12pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_12pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "4",
                "-gpu_ranks",
                "0",
                "1",
                "2",
                "3",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:1",
                "0:1",
                "0:1",
                "0:2",
                "0:2",
                "0:2",
                "0:3",
                "0:3",
                "0:3",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(120)
    def test_training_4gpus_24pairs(self):
        out_model_prefix = "wmt_4gpus_24pairs"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_24pairs.yml",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "4",
                "-gpu_ranks",
                "0",
                "1",
                "2",
                "3",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:1",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
                "0:2",
                "0:3",
                "0:3",
                "0:3",
                "0:3",
                "0:3",
                "0:3",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    @timeout_decorator.timeout(120)
    def test_training_1gpu_tensorboard(self):
        out_model_prefix = "wmt_1gpu_tb"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-tensorboard",
                "-tensorboard_log_dir",
                "tensorboard/{}".format(out_model_prefix),
                "-report_stats_from_parameters",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "1",
                "-gpu_ranks",
                "0",
                "-node_gpu",
                "0:0",
                "0:0",
                "0:0",
                "0:0",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        self.assertIn(out_model_prefix, os.listdir("tensorboard"))
        self.assertEqual(1, len(os.listdir("tensorboard/{}".format(out_model_prefix))))
        self.assertEqual("-rank0:0", os.listdir("tensorboard/{}".format(out_model_prefix))[0][-8:])
        if os.path.exists("tensorboard/{}".format(out_model_prefix)):
            logger.info("Removing folder {}".format("tensorboard/{}".format(out_model_prefix)))
            shutil.rmtree("tensorboard/{}".format(out_model_prefix))

    @timeout_decorator.timeout(120)
    def test_training_2gpus_tensorboard(self):
        out_model_prefix = "wmt_2gpus_tb"
        opts, _ = self.parser.parse_known_args(
            [
                "-config",
                "config/wmt_4pairs.yml",
                "-tensorboard",
                "-tensorboard_log_dir",
                "tensorboard/{}".format(out_model_prefix),
                "-report_stats_from_parameters",
                "-save_model",
                "models/{}".format(out_model_prefix),
                "-world_size",
                "2",
                "-gpu_ranks",
                "0",
                "1",
                "-node_gpu",
                "0:0",
                "0:1",
                "0:0",
                "0:1",
            ]
        )
        components = self._get_model_components(opts)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opts)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        self.assertIn(out_model_prefix, os.listdir("tensorboard"))
        tensorboard_run_folders = sorted(os.listdir("tensorboard/{}".format(out_model_prefix)), key=lambda x: x[::-1])
        self.assertEqual(2, len(os.listdir("tensorboard/{}".format(out_model_prefix))))
        self.assertEqual("-rank0:0", tensorboard_run_folders[0][-8:])
        self.assertEqual("-rank0:1", tensorboard_run_folders[1][-8:])
        if os.path.exists("tensorboard/{}".format(out_model_prefix)):
            logger.info("Removing folder {}".format("tensorboard/{}".format(out_model_prefix)))
            shutil.rmtree("tensorboard/{}".format(out_model_prefix))


# class TestTranslate(TestCase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.parser = ArgumentParser(description="translate.py")
#         mammoth.opts.config_opts(cls.parser)
#         mammoth.opts.translate_opts(cls.parser)
#         mammoth.opts.build_bilingual_model(cls.parser)
#
#     def test_translate(self):
#         # TODO: train model instead of loading one the one used now,
#         # remove all absolute paths, add test data in the repo
#         opts, _ = self.parser.parse_known_args(
#             [
#                 "-gpu",
#                 "0",
#                 "-data_type",
#                 "text",
#                 "-src_lang",
#                 "en",
#                 "-tgt_lang",
#                 "fr",
#                 "-model",
#                 "/home/micheleb/models/scaleUpMNMT/opus12/opus12.50.adaf_step_3000_",
#                 "-src",
#                 "/home/micheleb/data/scaleUpMNMT/prepare_opus_data_out/supervised/en-fr/opus.en-fr-dev.en.sp",
#                 "-output",
#                 "/home/micheleb/projects/OpenNMT-py-v2/translate_test.tmp",
#                 "-use_attention_bridge",
#             ]
#         )
#         translate(opts)
