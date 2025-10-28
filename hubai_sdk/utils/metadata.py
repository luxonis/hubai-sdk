from dataclasses import dataclass
from pathlib import Path

import onnx

from hubai_sdk.utils.types import DataType


@dataclass
class Metadata:
    input_shapes: dict[str, list[int]]
    input_dtypes: dict[str, DataType]
    output_shapes: dict[str, list[int]]
    output_dtypes: dict[str, DataType]


def get_metadata(model_path: Path) -> Metadata:
    suffix = model_path.suffix
    if suffix == ".onnx":
        return _get_metadata_onnx(model_path)
    if suffix in {".xml", ".bin"}:
        if suffix == ".xml":
            xml_path = model_path
            bin_path = model_path.with_suffix(".bin")
        else:
            bin_path = model_path
            xml_path = model_path.with_suffix(".xml")
        return _get_metadata_ir(bin_path, xml_path)
    if suffix == ".tflite":
        return _get_metadata_tflite(model_path)
    raise ValueError(f"Unsupported model format: {suffix}")


def _get_metadata_ir(bin_path: Path, xml_path: Path) -> Metadata:
    from openvino.runtime import Core

    ie = Core()
    try:
        model = ie.read_model(model=str(xml_path), weights=str(bin_path))
    except Exception as e:
        raise ValueError(
            f"Failed to load IR model: `{bin_path}` and `{xml_path}`"
        ) from e

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}

    for inp in model.inputs:
        name = next(iter(inp.names))
        input_shapes[name] = list(inp.shape)
        input_dtypes[name] = DataType.from_ir_runtime_dtype(
            inp.element_type.get_type_name()
        )
    for output in model.outputs:
        name = next(iter(output.names))
        output_shapes[name] = list(output.shape)
        output_dtypes[name] = DataType.from_ir_runtime_dtype(
            output.element_type.get_type_name()
        )

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _get_metadata_onnx(onnx_path: Path) -> Metadata:
    try:
        model = onnx.load(str(onnx_path))
    except Exception as e:
        raise ValueError(f"Failed to load ONNX model: `{onnx_path}`") from e

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}

    for inp in model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        input_shapes[inp.name] = shape
        input_dtypes[inp.name] = DataType.from_onnx_dtype(
            inp.type.tensor_type.elem_type
        )

    for output in model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        output_shapes[output.name] = shape
        output_dtypes[output.name] = DataType.from_onnx_dtype(
            output.type.tensor_type.elem_type
        )

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )


def _get_metadata_tflite(model_path: Path) -> Metadata:
    import tflite

    with open(model_path, "rb") as f:
        data = f.read()

    subgraph = tflite.Model.GetRootAsModel(data, 0).Subgraphs(0)

    if subgraph is None:
        raise ValueError("Failed to load TFLite model.")

    input_shapes = {}
    input_dtypes = {}
    output_shapes = {}
    output_dtypes = {}

    for i in range(subgraph.InputsLength()):
        tensor = subgraph.Tensors(subgraph.Inputs(i))
        input_shapes[tensor.Name().decode("utf-8")] = (  # type: ignore
            tensor.ShapeAsNumpy().tolist()  # type: ignore
        )
        input_dtypes[tensor.Name().decode("utf-8")] = (  # type: ignore
            DataType.from_tensorflow_dtype(tensor.Type())  # type: ignore
        )

    for i in range(subgraph.OutputsLength()):
        tensor = subgraph.Tensors(subgraph.Outputs(i))
        output_shapes[tensor.Name().decode("utf-8")] = (  # type: ignore
            tensor.ShapeAsNumpy().tolist()  # type: ignore
        )
        output_dtypes[tensor.Name().decode("utf-8")] = (  # type: ignore
            DataType.from_tensorflow_dtype(tensor.Type())  # type: ignore
        )

    return Metadata(
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )
