from dataclasses import dataclass
from pathlib import Path

from luxonis_ml.typing import PathType

from hubai_sdk.typing import QuantizationData, QuantizationInputType


@dataclass(frozen=True)
class NormalizedQuantizationInput:
    quantization_data: str | None
    custom_zip_path: Path | None
    input_type: QuantizationInputType


def is_custom_quantization_zip_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".zip"


def is_gcs_path(path: str) -> bool:
    return path.lower().startswith("gs://")


def normalize_quantization_input(
    quantization_data: QuantizationData | PathType | None,
) -> NormalizedQuantizationInput:
    if isinstance(quantization_data, Path):
        if not is_custom_quantization_zip_path(quantization_data):
            raise ValueError(
                "`quantization_data` paths must point to a .zip file."
            )
        quantization_data = str(quantization_data)

    if isinstance(quantization_data, str) and is_custom_quantization_zip_path(
        quantization_data
    ):
        custom_zip_path = Path(quantization_data).expanduser()
        if not custom_zip_path.exists():
            raise FileNotFoundError(
                f"Quantization zip not found: {custom_zip_path}"
            )
        return NormalizedQuantizationInput(
            quantization_data="CUSTOM",
            custom_zip_path=custom_zip_path,
            input_type="custom_zip",
        )

    if quantization_data is None:
        return NormalizedQuantizationInput(
            quantization_data=None,
            custom_zip_path=None,
            input_type="none",
        )

    if is_gcs_path(quantization_data):
        return NormalizedQuantizationInput(
            quantization_data=quantization_data,
            custom_zip_path=None,
            input_type="gcs_path",
        )

    normalized_quantization_data = quantization_data.upper()
    if normalized_quantization_data == "CUSTOM":
        raise ValueError(
            "`quantization_data='CUSTOM'` is not enough on its own. Pass the .zip path as `quantization_data`."
        )

    if quantization_data.startswith("aid_"):
        return NormalizedQuantizationInput(
            quantization_data=quantization_data,
            custom_zip_path=None,
            input_type="dataset_id",
        )

    return NormalizedQuantizationInput(
        quantization_data=normalized_quantization_data,
        custom_zip_path=None,
        input_type="predefined_dataset",
    )
