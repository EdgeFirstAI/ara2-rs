"""
EdgeFirst ARA-2 Python Library

Python bindings for the ARA-2 neural accelerator client library,
providing efficient NPU inference from Python via a proxy service
running on NXP i.MX platforms with Kinara ARA-2 hardware.

The actual implementation is in Rust via PyO3.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import numpy.typing as npt

__version__: str

DEFAULT_SOCKET: str
"""Default UNIX socket path for the ARA-2 proxy ("/var/run/ara2.sock")."""

# =============================================================================
# Exceptions
# =============================================================================

class Ara2Error(RuntimeError):
    """Base exception for all ARA-2 errors."""
    ...

class LibraryError(Ara2Error):
    """Failed to load libaraclient.so.1."""
    ...

class HardwareError(Ara2Error):
    """NPU hardware fault or endpoint state error."""
    ...

class ProxyError(Ara2Error):
    """Proxy connection or communication error."""
    ...

class ModelError(Ara2Error):
    """Model loading or inference error."""
    ...

class TensorError(Ara2Error):
    """Tensor allocation, shape, or DMA-BUF error."""
    ...

class MetadataError(Ara2Error):
    """DVM metadata parsing error (ZIP/JSON)."""
    ...

# =============================================================================
# Enums
# =============================================================================

class State:
    """Endpoint operational state returned by ``Endpoint.check_status()``."""

    Init: State
    """Device is initializing. Not yet ready for model loading."""
    Idle: State
    """Device is idle and ready to load models or run inference."""
    Active: State
    """Device is actively running inference at normal clock speed."""
    ActiveSlow: State
    """Device is running inference at reduced clock speed (power saving)."""
    ActiveBoosted: State
    """Device is running inference at boosted clock speed."""
    ThermalInactive: State
    """Device suspended due to thermal limits. Wait for it to cool."""
    ThermalUnknown: State
    """Thermal state cannot be determined."""
    Inactive: State
    """Device is powered down or not available."""
    Fault: State
    """Unrecoverable hardware error. Restart ara2-proxy."""

    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ModelOutputType:
    """Type of output produced by a model layer."""

    Classification: ModelOutputType
    Detection: ModelOutputType
    SemanticSegmentation: ModelOutputType
    Raw: ModelOutputType

    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# =============================================================================
# Data Classes
# =============================================================================

class DramStatistics:
    """DRAM usage statistics for an endpoint.

    All sizes are in bytes.
    """

    dram_size: int
    """Total DRAM capacity in bytes."""
    dram_occupancy_size: int
    """Total occupied DRAM in bytes."""
    free_size: int
    """Free DRAM in bytes."""
    reserved_occupancy_size: int
    """Reserved DRAM in bytes."""
    model_occupancy_size: int
    """DRAM occupied by loaded models in bytes."""
    tensor_occupancy_size: int
    """DRAM occupied by tensor buffers in bytes."""

class ModelTiming:
    """Timing information from a model inference run."""

    run_time_us: int
    """NPU inference execution time in microseconds."""
    input_time_us: int
    """Input DMA transfer time in microseconds."""
    output_time_us: int
    """Output DMA transfer time in microseconds."""

class InputQuantization:
    """Input tensor quantization parameters.

    ARA-2 input tensors are quantized to uint8 (or int8 for signed).
    The float-to-int conversion is: ``int_val = float_val * qn``.
    """

    qn: float
    """Quantization multiplier. ``int_val = float_val * qn``."""
    scale: float
    """Preprocessing scale factor."""
    mean: float
    """Per-tensor mean subtracted during preprocessing."""
    is_signed: bool
    """True if the tensor uses signed int8, False for uint8."""

class OutputQuantization:
    """Output tensor quantization parameters.

    Use ``Model.dequantize()`` for automatic conversion, or apply
    manually: ``float_val = int_val / qn`` (signed: treat byte as int8 first).
    """

    qn: float
    """Quantization divisor. ``float_val = int_val / qn``."""
    scale: float
    """Output scale factor."""
    offset: int
    """Zero-point offset."""
    is_signed: bool
    """True if the tensor uses signed int8, False for uint8."""

class InputTensorInfo:
    """Detailed information about an input tensor."""

    layer_id: int
    blob_id: int
    layer_name: str
    blob_name: str
    layer_type: str
    layout: str
    """Data layout string (e.g., "NCHW")."""
    size: int
    """Total size in bytes."""
    width: int
    height: int
    nch: int
    """Number of channels."""
    bpp: int
    """Bytes per element."""
    batch_size: int
    quant: InputQuantization

class OutputTensorInfo:
    """Detailed information about an output tensor."""

    layer_id: int
    blob_id: int
    fused_parent_id: int
    layer_name: str
    blob_name: str
    layer_fused_parent_name: str
    layer_type: str
    layout: str
    """Data layout string (e.g., "NCHW")."""
    size: int
    """Total size in bytes."""
    width: int
    height: int
    nch: int
    """Number of channels."""
    bpp: int
    """Bytes per element."""
    num_classes: int
    layer_output_type: ModelOutputType
    max_dynamic_id: int
    quant: OutputQuantization

# =============================================================================
# Core Classes
# =============================================================================

class Session:
    """ARA-2 session for communicating with the proxy.

    A Session represents a connection to the ARA-2 proxy service.
    Supports the context manager protocol for resource management::

        with Session.create_via_unix_socket("/var/run/ara2.sock") as session:
            endpoints = session.list_endpoints()

    Example::

        session = Session.create_via_unix_socket("/var/run/ara2.sock")
        versions = session.versions()
        endpoints = session.list_endpoints()
    """

    @staticmethod
    def create_via_unix_socket(socket_path: str) -> Session:
        """Create a session connected via UNIX domain socket.

        Args:
            socket_path: Path to the UNIX socket (e.g., "/var/run/ara2.sock")

        Raises:
            ProxyError: If the socket does not exist or the proxy is not running.
        """
        ...

    @staticmethod
    def create_via_tcp_ipv4_socket(ip: str, port: int) -> Session:
        """Create a session connected via TCP/IPv4 socket.

        Args:
            ip: IPv4 address as a string (e.g., "127.0.0.1")
            port: Port number

        Raises:
            ProxyError: If the address is unreachable or the proxy is not running.
            Ara2Error: If ``ip`` is not a valid IPv4 address string.
        """
        ...

    def versions(self) -> dict[str, str]:
        """Get version information for all components.

        Returns:
            Dictionary mapping component names to version strings.
        """
        ...

    def list_endpoints(self) -> list[Endpoint]:
        """List all available ARA-2 endpoints."""
        ...

    @property
    def socket_type(self) -> Literal["unix", "tcp"]:
        """Socket type used for this connection."""
        ...

    def __enter__(self) -> Session: ...
    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class Endpoint:
    """ARA-2 accelerator endpoint.

    An endpoint represents a single ARA-2 accelerator device that can
    load and execute neural network models.
    """

    def check_status(self) -> State:
        """Check the current status/state of the endpoint.

        Raises:
            HardwareError: If the endpoint is in a fault state.
        """
        ...

    def dram_statistics(self) -> DramStatistics:
        """Get DRAM statistics for the endpoint."""
        ...

    def load_model(self, model_path: str | os.PathLike[str]) -> Model:
        """Load a neural network model from a .dvm file.

        Args:
            model_path: Path to the compiled model file (.dvm)

        Raises:
            ModelError: If the file is not a valid DVM or is incompatible.
            HardwareError: If the endpoint has insufficient DRAM.
        """
        ...

    def __repr__(self) -> str: ...

class Model:
    """Neural network model loaded on an endpoint.

    Supports the context manager protocol to guarantee NPU resources
    are released::

        with endpoint.load_model("model.dvm") as model:
            model.allocate_tensors("dma")
            model.set_input_tensor(0, input_data)
            timing = model.run()
            output = model.get_output_tensor(0)

    For zero-copy preprocessing with edgefirst-hal::

        fd = model.input_tensor_fd(0)
        dst = hal.import_image(fd, w, h, hal.PixelFormat.PlanarRgb)
        processor.convert(src, dst)
        model.run()
    """

    # -- Lifecycle --

    def allocate_tensors(self, memory: str | None = None) -> None:
        """Allocate input and output tensors.

        Must be called before ``run()``, ``set_input_tensor()``, or any
        tensor accessor method.

        Args:
            memory: Memory type — ``"dma"``, ``"shm"``, ``"mem"``, or
                    ``None`` (auto-select, tries DMA first). Use ``"dma"``
                    for zero-copy workflows with edgefirst-hal.
        """
        ...

    def set_timeout_ms(self, timeout_ms: int) -> None:
        """Set the inference timeout in milliseconds (default: 1000)."""
        ...

    def run(self) -> ModelTiming:
        """Run inference on the model.

        Raises:
            TensorError: If tensors have not been allocated.
            ModelError: If inference times out (increase with ``set_timeout_ms``).
            HardwareError: If the NPU is in a fault or thermal state.
        """
        ...

    # -- Tensor I/O (numpy) --

    def set_input_tensor(self, index: int, data: npt.NDArray[np.uint8]) -> None:
        """Copy numpy array data into an input tensor.

        Args:
            index: Input tensor index (0-based)
            data: numpy uint8 array. Total byte count must match
                  ``input_size(index)``.

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or sizes don't match.
        """
        ...

    def get_output_tensor(self, index: int) -> npt.NDArray[np.uint8]:
        """Get output tensor data as a flat numpy uint8 array.

        Args:
            index: Output tensor index (0-based)

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated.
        """
        ...

    def dequantize(self, index: int) -> npt.NDArray[np.float32]:
        """Dequantize an output tensor to float32.

        Applies ``float_val = int_val / qn`` (signed: treats byte as int8).

        Args:
            index: Output tensor index (0-based)

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or qn is zero.
        """
        ...

    # -- DMA-BUF zero-copy --

    def input_tensor_fd(self, index: int) -> int:
        """Get a cloned DMA-BUF file descriptor for an input tensor.

        The returned FD is owned by the caller. Pass it to
        ``edgefirst_hal.import_image()`` for zero-copy GPU preprocessing.
        Close with ``os.close(fd)`` when done.

        Args:
            index: Input tensor index (0-based)

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or use system memory.
        """
        ...

    def output_tensor_fd(self, index: int) -> int:
        """Get a cloned DMA-BUF file descriptor for an output tensor.

        The returned FD is owned by the caller. Close with ``os.close(fd)``
        when done.

        Args:
            index: Output tensor index (0-based)

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or use system memory.
        """
        ...

    def input_tensor_memory(self, index: int) -> str:
        """Get the memory type of an input tensor: "dma", "shm", or "mem"."""
        ...

    def output_tensor_memory(self, index: int) -> str:
        """Get the memory type of an output tensor: "dma", "shm", or "mem"."""
        ...

    # -- Introspection --

    @property
    def n_inputs(self) -> int:
        """Number of input tensors."""
        ...

    @property
    def n_outputs(self) -> int:
        """Number of output tensors."""
        ...

    def input_shape(self, index: int) -> tuple[int, int, int]:
        """Input tensor shape as (channels, height, width)."""
        ...

    def output_shape(self, index: int) -> tuple[int, int, int]:
        """Output tensor shape as (channels, height, width)."""
        ...

    def input_size(self, index: int) -> int:
        """Input tensor size in bytes."""
        ...

    def output_size(self, index: int) -> int:
        """Output tensor size in bytes."""
        ...

    def input_bpp(self, index: int) -> int:
        """Bytes per element for an input tensor."""
        ...

    def output_bpp(self, index: int) -> int:
        """Bytes per element for an output tensor."""
        ...

    def input_info(self, index: int) -> InputTensorInfo:
        """Get detailed information about an input tensor."""
        ...

    def output_info(self, index: int) -> OutputTensorInfo:
        """Get detailed information about an output tensor."""
        ...

    def input_quants(self, index: int) -> InputQuantization:
        """Get quantization parameters for an input tensor."""
        ...

    def output_quants(self, index: int) -> OutputQuantization:
        """Get quantization parameters for an output tensor."""
        ...

    def __enter__(self) -> Model: ...
    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> bool: ...
    def __repr__(self) -> str: ...

# =============================================================================
# DVM Metadata
# =============================================================================

def read_metadata(path: str | os.PathLike[str]) -> DvmMetadata | None:
    """Read EdgeFirst metadata from a DVM model file.

    Returns None if the file has no embedded metadata.
    """
    ...

def read_labels(path: str | os.PathLike[str]) -> list[str]:
    """Read class labels from a DVM model file.

    Returns an empty list if no labels are present.
    """
    ...

def has_metadata(path: str | os.PathLike[str]) -> bool:
    """Check if a DVM file has embedded metadata."""
    ...

class DvmMetadata:
    """EdgeFirst metadata embedded in a DVM model file."""

    @property
    def task(self) -> str | None:
        """Model task type (e.g., "detect", "segment", "classify")."""
        ...
    @property
    def classes(self) -> list[str]:
        """Class labels from the dataset."""
        ...
    @property
    def dataset(self) -> DatasetInfo | None: ...
    @property
    def input(self) -> InputSpec | None: ...
    @property
    def model(self) -> ModelInfo | None: ...
    @property
    def deployment(self) -> DeploymentInfo | None: ...
    @property
    def compilation(self) -> CompilationInfo | None: ...
    @property
    def decoder_version(self) -> str | None:
        """Decoder version string (e.g., "yolov8")."""
        ...
    @property
    def nms(self) -> str | None:
        """NMS type (e.g., "class_agnostic")."""
        ...
    @property
    def outputs(self) -> list[OutputSpec]: ...

class DatasetInfo:
    """Training dataset information embedded in the model file."""

    @property
    def classes(self) -> list[str]:
        """Class label strings."""
        ...
    @property
    def id(self) -> str | None:
        """Dataset identifier (e.g., "coco2017")."""
        ...
    @property
    def name(self) -> str | None:
        """Human-readable dataset name."""
        ...

class ModelInfo:
    """Source model information embedded in the model file."""

    @property
    def model_task(self) -> str | None:
        """Task type (e.g., "detect", "segment", "classify")."""
        ...
    @property
    def model_size(self) -> str | None:
        """Model size variant (e.g., "s", "m", "l", "x")."""
        ...
    @property
    def model_version(self) -> str | None:
        """Source model version string."""
        ...
    @property
    def detection(self) -> bool:
        """True if the model produces detection outputs."""
        ...
    @property
    def segmentation(self) -> bool:
        """True if the model produces segmentation outputs."""
        ...

class DeploymentInfo:
    """Deployment metadata describing how this model was packaged."""

    @property
    def model_name(self) -> str | None: ...
    @property
    def name(self) -> str | None:
        """Human-readable deployment name."""
        ...
    @property
    def author(self) -> str | None: ...
    @property
    def description(self) -> str | None: ...

class InputSpec:
    """Input specification from model metadata."""

    @property
    def size(self) -> str | None:
        """Size string (e.g., "640x480")."""
        ...
    @property
    def input_channels(self) -> int | None: ...
    @property
    def output_channels(self) -> int | None: ...
    @property
    def cameraadaptor(self) -> str | None:
        """Camera adaptor type (e.g., "rgb", "bgr")."""
        ...
    def dimensions(self) -> tuple[int, int] | None:
        """Parse size string into (width, height)."""
        ...

class OutputSpec:
    """Output tensor specification from model metadata."""

    @property
    def index(self) -> int | None: ...
    @property
    def name(self) -> str | None: ...
    @property
    def output_type(self) -> str | None: ...
    @property
    def decoder(self) -> str | None: ...
    @property
    def decode(self) -> bool: ...
    @property
    def dtype(self) -> str | None: ...
    @property
    def shape(self) -> list[int]: ...

class CompilationInfo:
    """Compilation information from the DVM build process."""

    @property
    def target(self) -> str | None:
        """Target hardware (e.g., "ara-2")."""
        ...
    @property
    def format(self) -> str | None:
        """Model format (e.g., "dvm")."""
        ...
    @property
    def ppa(self) -> PpaMetrics | None: ...

class PpaMetrics:
    """Performance, power, and area metrics from compilation."""

    @property
    def ips(self) -> float | None:
        """Inferences per second."""
        ...
    @property
    def power_mw(self) -> float | None:
        """Power consumption in milliwatts."""
        ...
    @property
    def cycles(self) -> int | None:
        """Execution cycles."""
        ...
    @property
    def ddr_bw_mbps(self) -> float | None:
        """DDR bandwidth in MB/s."""
        ...

__all__ = [
    # Constants
    "DEFAULT_SOCKET",
    # Exceptions
    "Ara2Error",
    "LibraryError",
    "HardwareError",
    "ProxyError",
    "ModelError",
    "TensorError",
    "MetadataError",
    # Enums
    "State",
    "ModelOutputType",
    # Data classes
    "DramStatistics",
    "ModelTiming",
    "InputQuantization",
    "OutputQuantization",
    "InputTensorInfo",
    "OutputTensorInfo",
    # Core classes
    "Session",
    "Endpoint",
    "Model",
    # Metadata
    "read_metadata",
    "read_labels",
    "has_metadata",
    "DvmMetadata",
    "DatasetInfo",
    "ModelInfo",
    "DeploymentInfo",
    "InputSpec",
    "OutputSpec",
    "CompilationInfo",
    "PpaMetrics",
]
