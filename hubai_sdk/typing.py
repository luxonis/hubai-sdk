"""Public literal types accepted by HubAI service functions.

These aliases enumerate values sent to the HubAI API. They are useful when
building typed wrappers or configuration models; ordinary callers may pass the
corresponding strings directly.

Important aliases:
    - `Task` describes the problem solved by a model.
    - `License` contains license names accepted for model metadata.
    - `QuantizationMode` selects the precision and calibration strategy.
    - `QuantizationData` accepts a built-in domain or an ``aid_`` dataset ID.
    - `YoloVersion` selects the YOLO output decoder contract.

`Visibility` represents the API's three string labels: ``"public"``,
``"private"``, and ``"team"``. Public resource functions currently expose an
``is_public`` flag instead, and the meaning of ``None`` depends on the
operation: it creates a team-visible model, disables visibility filtering in
list operations, and leaves visibility unchanged in `update_model`.
"""

from typing import Annotated, Literal, TypeAlias

from pydantic import Field

Task: TypeAlias = Literal[
    "CLASSIFICATION",
    "OBJECT_DETECTION",
    "SEGMENTATION",
    "KEYPOINT_DETECTION",
    "DEPTH_ESTIMATION",
    "LINE_DETECTION",
    "FEATURE_DETECTION",
    "DENOISING",
    "LOW_LIGHT_ENHANCEMENT",
    "SUPER_RESOLUTION",
    "REGRESSION",
    "INSTANCE_SEGMENTATION",
    "IMAGE_EMBEDDING",
]

License: TypeAlias = Literal[
    "undefined",
    "MIT",
    "GNU General Public License v3.0",
    "GNU Affero General Public License v3.0",
    "Apache 2.0",
    "NTU S-Lab 1.0",
    "Ultralytics Enterprise",
    "CreativeML Open RAIL-M",
    "BSD 3-Clause",
]

Visibility: TypeAlias = Literal[
    "public",
    "private",
    "team",
]

Order: TypeAlias = Literal[
    "asc",
    "desc",
]

ModelClass: TypeAlias = Literal[
    "base",
    "exported",
]

Status: TypeAlias = Literal["available", "unavailable"]

TargetPrecision: TypeAlias = Literal["FP16", "FP32", "INT8", "INT8_INT16"]

QuantizationMode: TypeAlias = Literal[
    "INT8_STANDARD",
    "INT8_ACCURACY_FOCUSED",
    "INT8_INT16_MIXED",
    "INT8_INT16_MIXED_ACCURACY_FOCUSED",
    "FP16_STANDARD",
    "FP32_STANDARD",
]

Quantization: TypeAlias = Literal[
    "DRIVING",
    "FOOD",
    "GENERAL",
    "INDOORS",
    "RANDOM",
    "WAREHOUSE",
    "CLIP",
    "CUSTOM",
    "UNKNOWN",
]

DatasetId = Annotated[str, Field(pattern=r"^aid_[a-zA-Z0-9_]+")]

QuantizationData: TypeAlias = Quantization | DatasetId

YoloVersion: TypeAlias = Literal[
    "yolov5",
    "yolov6r1",
    "yolov6r3",
    "yolov6r4",
    "yolov7",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolov11",
    "yolov12",
    "yolov26",
    "goldyolo",
]

HubService: TypeAlias = Literal["models", "dags", "jobs"]
