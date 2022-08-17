import ast
import logging
import os
import shutil
from typing import List
from unittest import TestCase

import onmt
from onmt.bin.train import train
from onmt.bin.translate import translate
from onmt.utils.parse import ArgumentParser

logger = logging.getLogger(__name__)


class TestTraining(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = ArgumentParser(description="train.py")
        onmt.opts.train_opts(cls.parser)

    @staticmethod
    def _get_model_components(opt) -> List[str]:
        components_enc = ["{}_enc".format(src_lang) for src_lang in ast.literal_eval(opt.src_vocab).keys()]
        components_dec = ["{}_dec".format(tgt_lang) for tgt_lang in ast.literal_eval(opt.tgt_vocab).keys()]
        components_gen = ["{}_gen".format(tgt_lang) for tgt_lang in ast.literal_eval(opt.tgt_vocab).keys()]
        return ["frame", "bridge", *components_enc, *components_dec, *components_gen]

    def test_training_1gpu_4pairs(self):
        out_model_prefix = "wmt_1gpu_4pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_1gpu_4pairs_ab_lin(self):
        out_model_prefix = "wmt_1gpu_4pairs_lin"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_1gpu_4pairs_ab_ff(self):
        out_model_prefix = "wmt_1gpu_4pairs_ff"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_1gpu_4pairs_ab_tf(self):
        out_model_prefix = "wmt_1gpu_4pairs_tf"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_1gpu_4pairs_ab_simple(self):
        out_model_prefix = "wmt_1gpu_4pairs_simple"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_1gpu_4pairs_ab_perceiver(self):
        out_model_prefix = "wmt_1gpu_4pairs_perceiver"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs(self):
        out_model_prefix = "wmt_2gpus_4pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs_ab_lin(self):
        out_model_prefix = "wmt_2gpus_4pairs_lin"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs_ab_ff(self):
        out_model_prefix = "wmt_2gpus_4pairs_ff"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs_ab_tf(self):
        out_model_prefix = "wmt_2gpus_4pairs_tf"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs_ab_simple(self):
        out_model_prefix = "wmt_2gpus_4pairs_simple"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs_ab_perceiver(self):
        out_model_prefix = "wmt_2gpus_4pairs_perceiver"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_2gpus_4pairs_crossed(self):
        out_model_prefix = "wmt_2gpus_4pairs_crossed"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_4gpus_4pairs(self):
        out_model_prefix = "wmt_4gpus_4pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_3gpus_12pairs(self):
        out_model_prefix = "wmt_3gpus_12pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_3gpus_21pairs(self):
        out_model_prefix = "wmt_3gpus_21pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_4gpus_12pairs(self):
        out_model_prefix = "wmt_4gpus_12pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_4gpus_24pairs(self):
        out_model_prefix = "wmt_4gpus_24pairs"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)

    def test_training_1gpu_tensorboard(self):
        out_model_prefix = "wmt_1gpu_tb"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
        for cmp in components:
            self.assertNotIn("{}_step_2_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
            self.assertIn("{}_step_4_{}.pt".format(out_model_prefix, cmp), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        self.assertIn(out_model_prefix, os.listdir("tensorboard"))
        self.assertEqual(1, len(os.listdir("tensorboard/{}".format(out_model_prefix))))
        self.assertEqual("-rankNone:0", os.listdir("tensorboard/{}".format(out_model_prefix))[0][-11:])
        if os.path.exists("tensorboard/{}".format(out_model_prefix)):
            logger.info("Removing folder {}".format("tensorboard/{}".format(out_model_prefix)))
            shutil.rmtree("tensorboard/{}".format(out_model_prefix))

    def test_training_2gpus_tensorboard(self):
        out_model_prefix = "wmt_2gpus_tb"
        opt, _ = self.parser.parse_known_args(
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
        components = self._get_model_components(opt)
        out_files = ["models/{}_step_4_{}.pt".format(out_model_prefix, cmp) for cmp in components]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
            logger.info("Launch training")
        train(opt)
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


class TestTranslate(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = ArgumentParser(description="translate.py")
        onmt.opts.config_opts(cls.parser)
        onmt.opts.translate_opts(cls.parser)
        onmt.opts.build_bilingual_model(cls.parser)

    def test_translate(self):
        # TODO: train model instead of loading one the one used now,
        # remove all absolute paths, add test data in the repo
        opt, _ = self.parser.parse_known_args(
            [
                "-gpu",
                "0",
                "-data_type",
                "text",
                "-src_lang",
                "en",
                "-tgt_lang",
                "fr",
                "-model",
                "/home/micheleb/models/scaleUpMNMT/opus12/opus12.50.adaf_step_3000_",
                "-src",
                "/home/micheleb/data/scaleUpMNMT/prepare_opus_data_out/supervised/en-fr/opus.en-fr-dev.en.sp",
                "-output",
                "/home/micheleb/projects/OpenNMT-py-v2/translate_test.tmp",
                "-use_attention_bridge",
            ]
        )
        translate(opt)
