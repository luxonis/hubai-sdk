import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from loguru import logger
from luxonis_ml.nn_archive.config import Config as NNArchiveConfig
from luxonis_ml.nn_archive.config_building_blocks import (
    InputType,
)

from hubai_sdk.utils.config import Config
from hubai_sdk.utils.constants import MISC_DIR


def process_nn_archive(
    path: Path, overrides: dict[str, Any] | None
) -> tuple[Config, NNArchiveConfig, str]:
    """Extracts the archive from tar and parses its config.

    @type path: Path
    @param path: Path to the archive.
    @type overrides: Optional[Dict[str, Any]]
    @param overrides: Config overrides.
    @rtype: Tuple[Config, NNArchiveConfig, str]
    @return: Tuple of the parsed config, NNArchiveConfig and the main
        stage key.
    """

    untar_path = MISC_DIR / path.stem
    if path.is_dir():
        untar_path = path
    elif tarfile.is_tarfile(path):
        if untar_path.suffix == ".tar":
            untar_path = MISC_DIR / untar_path.stem

        def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
            """Filter members to prevent path traversal attacks."""
            safe_files = []
            for member in tar.getmembers():
                # Normalize path and ensure it's within the extraction folder
                if not member.name.startswith("/") and ".." not in member.name:
                    safe_files.append(member)
                else:
                    logger.warning(f"Skipping unsafe file: {member.name}")
            return safe_files

        with tarfile.open(path, mode="r") as tf:
            for member in safe_members(tf):
                tf.extract(member, path=untar_path)

    else:
        raise RuntimeError(f"Unknown NN Archive path: `{path}`")

    if not (untar_path / "config.json").exists():
        raise RuntimeError(f"NN Archive config not found in `{untar_path}`")

    with open(untar_path / "config.json") as f:
        archive_config = NNArchiveConfig(**json.load(f))

    main_stage_config = {
        "name": archive_config.model.metadata.name,
        "input_model": str(untar_path / archive_config.model.metadata.path),
        "inputs": [],
        "outputs": [],
    }

    for inp in archive_config.model.inputs:
        reverse = inp.preprocessing.reverse_channels
        interleaved_to_planar = inp.preprocessing.interleaved_to_planar
        dai_type = inp.preprocessing.dai_type

        layout = inp.layout
        encoding = "NONE"
        if inp.input_type == InputType.IMAGE:
            if dai_type is not None:
                if (reverse and dai_type.startswith("BGR")) or (
                    reverse is False and dai_type.startswith("RGB")
                ):
                    logger.warning(
                        "'reverse_channels' and 'dai_type' are conflicting, using dai_type"
                    )

                if dai_type.startswith("RGB"):
                    encoding = {"from": "RGB", "to": "BGR"}
                elif dai_type.startswith("BGR"):
                    encoding = "BGR"
                elif dai_type.startswith("GRAY"):
                    encoding = "GRAY"
                else:
                    logger.warning("unknown dai_type, using RGB888p")
                    encoding = {"from": "RGB", "to": "BGR"}

                if (interleaved_to_planar and dai_type.endswith("p")) or (
                    interleaved_to_planar is False and dai_type.endswith("i")
                ):
                    logger.warning(
                        "'interleaved_to_planar' and 'dai_type' are conflicting, using dai_type"
                    )
                if dai_type.endswith("i"):
                    layout = "NHWC"
                elif dai_type.endswith("p"):
                    layout = "NCHW"
            else:
                if reverse is not None:
                    logger.warning(
                        "'reverse_channels' flag is deprecated and will be removed in the future, use 'dai_type' instead"
                    )
                    if reverse:
                        encoding = {"from": "RGB", "to": "BGR"}
                    else:
                        encoding = "BGR"
                else:
                    encoding = {"from": "RGB", "to": "BGR"}

                if interleaved_to_planar is not None:
                    logger.warning(
                        "'interleaved_to_planar' flag is deprecated and will be removed in the future, use 'dai_type' instead"
                    )
                    if interleaved_to_planar:
                        layout = "NHWC"
                    else:
                        layout = "NCHW"
            channels = (
                inp.shape[layout.index("C")]
                if layout and "C" in layout
                else None
            )
            if channels and channels == 1:
                encoding = "GRAY"

        _enc = encoding if isinstance(encoding, str) else encoding["from"]
        mean = inp.preprocessing.mean
        if mean is None:
            if _enc in {"RGB", "BGR"}:
                mean = [0, 0, 0]
            elif _enc == "GRAY":
                mean = [0]

        scale = inp.preprocessing.scale
        if scale is None:
            if _enc in {"RGB", "BGR"}:
                scale = [1, 1, 1]
            elif _enc == "GRAY":
                scale = [1]

        main_stage_config["inputs"].append(
            {
                "name": inp.name,
                "shape": inp.shape,
                "layout": layout,
                "data_type": inp.dtype.value,
                "mean_values": mean,
                "scale_values": scale,
                "encoding": encoding
                if isinstance(encoding, dict)
                else {"from": encoding, "to": encoding},
            }
        )

    for out in archive_config.model.outputs:
        main_stage_config["outputs"].append(
            {
                "name": out.name,
                "shape": out.shape,
                "layout": out.layout,
                "data_type": out.dtype.value,
            }
        )

    stages = {}

    for head in archive_config.model.heads or []:
        postprocessor_path = getattr(head.metadata, "postprocessor_path", None)
        if postprocessor_path is not None:
            input_model_path = untar_path / postprocessor_path
            head_stage_config = {
                "input_model": str(input_model_path),
                "inputs": [],
                "outputs": [],
                "encoding": {"from": "NONE", "to": "NONE"},
            }
            stages[input_model_path.stem] = head_stage_config

    if stages:
        main_stage_key = main_stage_config.pop("name")
        config = {
            "name": main_stage_key,
            "stages": {
                main_stage_key: main_stage_config,
                **stages,
            },
        }
    else:
        config = main_stage_config
        main_stage_key = config["name"]

    return Config.get_config(config, overrides), archive_config, main_stage_key


def cleanup_extracted_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
