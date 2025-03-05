"""
 This file is from
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import warnings

from omegaconf import OmegaConf
import torch.distributed as dist

import pipeline.common.utils as utils
from pipeline.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from pipeline.common.registry import registry
from pipeline.processors.base_processor import BaseProcessor
from pipeline.datasets.datasets.multimodal_dataset import MultimodalDataset


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # only called on 1 GPU/TPU in distributed
        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        txt_proc_cfg = self.config.get("text_processor")

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        """Download text data if necessary"""
        if hasattr(self.config.build_info, "text_data"):
            text_data = self.config.build_info.text_data
            cache_root = registry.get_path("cache_root")

            for split, info in text_data.items():
                if not hasattr(info, "storage"):
                    continue
                    
                storage_path = info.storage
                if not os.path.isabs(storage_path):
                    storage_path = os.path.join(cache_root, storage_path)

                dirname = os.path.dirname(storage_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

    def build(self):
        """
        Create datasets by split 
        """
        self.build_processors()
        build_info = self.config.build_info
        
        datasets = dict()
        
        # Handle text data based on the configuration
        if hasattr(build_info, "text_data"):
            for split in build_info.text_data.keys():
                if split not in ["train", "val", "test"]:
                    continue

                is_train = split == "train"
                text_processor = (
                    self.text_processors["train"]
                    if is_train
                    else self.text_processors["eval"]
                )

                text_path = build_info.text_data.get(split).storage
                if not os.path.isabs(text_path):
                    text_path = utils.get_cache_path(text_path)

                dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
                datasets[split] = dataset_cls(
                    text_processor=text_processor,
                    text_path=text_path,
                )

        return datasets


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg


@registry.register_builder("text_dataset")
class BaseBuilder(BaseDatasetBuilder):
    train_dataset_cls = MultimodalDataset  # Assuming this works with text-only data

    DATASET_CONFIG_DICT = {
        "default": None,
    }

    def build(self):
        build_info = self.config.build_info

        datasets = dict()

        # create datasets based on storage paths
        dataset_cls = self.train_dataset_cls
        if hasattr(build_info, "storage") and build_info.storage is not None:
            datasets["train"] = dataset_cls(
                datapath=build_info.storage,
                is_train=True,
            )
        if hasattr(build_info, "storage_valid") and build_info.storage_valid is not None:
            datasets["valid"] = dataset_cls(
                datapath=build_info.storage_valid,
            )
        if hasattr(build_info, "storage_test") and build_info.storage_test is not None:
            datasets["test"] = dataset_cls(
                datapath=build_info.storage_test,
            )
        return datasets