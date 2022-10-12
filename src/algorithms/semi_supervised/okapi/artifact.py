from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Final, Optional, Union

import attr
from conduit.logging import init_logger
import torch
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.wandb_run import Run

from .matching import MatchedIndicesBD

__all__ = [
    "save_indices_artifact",
    "load_indices_from_artifact",
]

LOGGER = init_logger(__file__)


DEFAULT_FILENAME: Final[str] = "match_indices.pt"


def save_indices_artifact(
    indices: MatchedIndicesBD,
    *,
    run: Union[Run, RunDisabled],
    name: str = "real_patch",
    filename: str = DEFAULT_FILENAME,
    metadata: Optional[Dict[str, Any]] = None,
):
    artifact = wandb.Artifact(
        name,
        type="match_indices",
        description="Matched-indices per direction (treatment -> control, control -> treatment)",
        metadata=metadata,
    )
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        save_path = tmpdir / filename
        LOGGER.info(f"Match-indices saved to '{save_path.resolve()}'")
        state_dict = attr.asdict(indices)
        torch.save(state_dict, f=save_path)
        artifact.add_file(str(save_path.resolve()), name=filename)
        run.log_artifact(artifact)
        artifact.wait()
        LOGGER.info(
            "Match-indices artifact saved to "
            f"'{run.entity}/{run.project}/{name}:{artifact.version}'"
        )


def load_indices_from_artifact(
    name: str,
    *,
    run: Union[Run, RunDisabled],
    project: Optional[str] = None,
    filename: str = DEFAULT_FILENAME,
    root: Optional[Union[Path, str]] = None,
) -> MatchedIndicesBD:
    if root is None:
        root = Path("artifacts") / "match_indices"
    root = Path(root)
    if project is None:
        project = f"{run.entity}/{run.project}"
    full_name = f"{project}/{name}"
    artifact_dir = root / name
    filepath = artifact_dir / filename
    if not filepath.exists():
        LOGGER.info("Downloading match-indices artifact...")
        artifact = run.use_artifact(full_name)
        artifact.download(root=artifact_dir)
    state_dict = torch.load(filepath)
    LOGGER.info(f"Match-indices successfully loaded from artifact '{full_name}'.")
    return MatchedIndicesBD.from_dict(state_dict)
