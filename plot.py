import argparse
import os
import torch
from pytorch_lightning.loggers import WandbLogger
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)
from gp.lightning.metric import (
    flat_binary_func,
    EvalKit,
)
from gp.lightning.data_template import DataModule
from gp.lightning.training import lightning_fit
from gp.lightning.module_template import ExpConfig
from gp.lightning.metric import HitsAtK
from types import SimpleNamespace
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel
from models.model import PyGRGCNEdge

from torchmetrics import AUROC, Accuracy
from utils import (
    SentenceEncoder,
    MultiApr,
    MultiAuc,
    ENCODER_DIM_DICT,
)

from task_constructor import UnifiedTaskConstructor


def main(params):
    encoder = SentenceEncoder(params.llm_name, root=".", batch_size=params.llm_b_size)
    task_config_lookup = load_yaml(
        os.path.join(os.path.dirname(__file__), "configs", "task_config.yaml")
    )
    data_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "data_config.yaml"))

    if isinstance(params.task_names, str):
        task_names = [a.strip() for a in params.task_names.split(",")]
    else:
        task_names = params.task_names

    root = "cache_data"
    if params.llm_name != "ST":
        root = f"cache_data_{params.llm_name}"

    tasks = UnifiedTaskConstructor(
        task_names,
        encoder,
        task_config_lookup,
        data_config_lookup,
        root=root,
        batch_size=params.batch_size,
        sample_size=params.train_sample_size,
    )
    val_task_index_lst, val_pool_mode = tasks.construct_exp()
    # remove llm model
    encoder.flush_model()

    in_dim = ENCODER_DIM_DICT[params.llm_name]
    out_dim = 768 + (params.rwpe if params.rwpe is not None else 0)
    # out_dim = 768

    if hasattr(params, "d_multiple"):
        if isinstance(params.d_multiple, str):
            data_multiple = [float(a) for a in params.d_multiple.split(",")]
        else:
            data_multiple = params.d_multiple
    else:
        data_multiple = [1]

    if hasattr(params, "d_min_ratio"):
        if isinstance(params.d_min_ratio, str):
            min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
        else:
            min_ratio = params.d_min_ratio
    else:
        min_ratio = [1]

    train_data = tasks.make_train_data(data_multiple, min_ratio, data_val_index=val_task_index_lst)

    text_dataset = tasks.make_full_dm_list(
        data_multiple, min_ratio, train_data
    )

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    params = parser.parse_args()
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)
    # Add for few-shot parameters

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "full_cdm"
    print(params)
    main(params)
