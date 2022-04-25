import logging
import os
from unittest import TestCase

import onmt
from onmt.bin.train import train
from onmt.utils.parse import ArgumentParser

logger = logging.getLogger(__name__)


class TestTraining(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = ArgumentParser(description="train.py")
        onmt.opts.train_opts(cls.parser)

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
        out_file = "models/{}_0_step_20.pt".format(out_model_prefix)
        if os.path.exists(out_file):
            logger.info("Removing file {}".format(out_file))
            os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 2)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 2)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 4)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_2_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_3_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_2_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_3_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 3)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_2_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_2_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 3)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_2_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_2_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 4)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_2_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_3_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_2_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_3_step_20.pt".format(out_model_prefix), os.listdir("models"))
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
        out_files = ["models/{}_{}_step_20.pt".format(out_model_prefix, gpu_rank) for gpu_rank in range(0, 4)]
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
        logger.info("Launch training")
        train(opt)
        self.assertNotIn("{}_0_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_1_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_2_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertNotIn("{}_3_step_10.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_0_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_1_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_2_step_20.pt".format(out_model_prefix), os.listdir("models"))
        self.assertIn("{}_3_step_20.pt".format(out_model_prefix), os.listdir("models"))
        for out_file in out_files:
            if os.path.exists(out_file):
                logger.info("Removing file {}".format(out_file))
                os.remove(out_file)
