from __future__ import annotations
import os

from ranzen.hydra import Option
import torch.multiprocessing

from src.algorithms import Erm
from src.algorithms.self_supervised import Moco, SimCLR
from src.algorithms.semi_supervised import FixMatch, Okapi, OkapiOffline
from src.data.datamodules import (
    Camelyon17DataModule,
    CivilCommentsDataModule,
    FMoWDataModule,
    GlobalWheatDatamodule,
    IWildCamDataModule,
    PovertyMapDataModule,
)
from src.models.artifact import ArtifactLoader
from src.models.backbones import (
    Beit,
    Bert,
    Clip,
    ConvNeXt,
    DistilBert,
    RegNet,
    ResNet,
    Swin,
    ViT,
)
from src.models.meta import LinearProbe
from src.models.meta.ema import EmaModel
from src.models.meta.ft import BitFit
from src.models.predictors import Fcn
from src.relays import WILDSRelay

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    dm_ops: list[Option] = [
        Option(Camelyon17DataModule, name="camelyon17"),
        Option(CivilCommentsDataModule, name="civil_comments"),
        Option(FMoWDataModule, name="fmow"),
        Option(GlobalWheatDatamodule, name="global_wheat"),
        Option(IWildCamDataModule, name="iwildcam"),
        Option(PovertyMapDataModule, name="poverty_map"),
    ]
    alg_ops: list[Option] = [
        Option(Erm, "erm"),
        Option(FixMatch, "fixmatch"),
        Option(Moco, "moco"),
        Option(OkapiOffline, "okapi_off"),
        Option(Okapi, "okapi"),
        Option(SimCLR, "simclr"),
    ]
    bb_ops: list[Option] = [
        Option(Beit, "beit"),
        Option(Clip, "clip"),
        Option(ConvNeXt, "convnext"),
        Option(RegNet, "regnet"),
        Option(ResNet, "resnet"),
        Option(Swin, "swin"),
        Option(ViT, "vit"),
        Option(Bert, "bert"),
        Option(DistilBert, "dbert"),
        Option(ArtifactLoader, "artifact"),
    ]

    pred_ops: list[Option] = [
        Option(Fcn, "fcn"),
    ]
    mm_ops: list[Option] = [
        Option(BitFit, "bitfit"),
        Option(EmaModel, "ema"),
        Option(LinearProbe, "lp"),
    ]

    WILDSRelay.with_hydra(
        root="conf",
        dm=dm_ops,
        alg=alg_ops,
        backbone=bb_ops,
        predictor=pred_ops,
        meta_model=mm_ops,
        clear_cache=True,
    )
