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

TargetPrecision: TypeAlias = Literal["FP16", "FP32", "INT8"]

Quantization: TypeAlias = Literal[
    "DRIVING", "FOOD", "GENERAL", "INDOORS", "RANDOM", "WAREHOUSE"
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
    "goldyolo",
]

HubService: TypeAlias = Literal["models", "dags"]
