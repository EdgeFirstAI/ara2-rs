"""
EdgeFirst ARA-2 Python Library

Python bindings for the ARA-2 neural accelerator client library,
providing efficient NPU inference from Python via a proxy service
running on NXP i.MX platforms with Kinara ARA-2 hardware.

The actual implementation is in Rust via PyO3.
"""

import os
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt

__version__: str

DEFAULT_SOCKET: str
"""Default UNIX socket path for the ARA-2 proxy ("/var/run/proxy.sock").

Matches ``interface_socket_file`` in NXP's rt-sdk-ara2
``proxy_config.yaml``, and is the first entry in :data:`SOCKET_PATHS`.
Override at runtime with the ``ARA2_SOCKET`` environment variable -- see
:func:`proxy_endpoint` and :meth:`Session.connect`.
"""

SOCKET_PATHS: list[str]
"""Proxy socket paths tried, in order, when ``ARA2_SOCKET`` is not set.

The socket path is a packaging decision and the two packagings disagree:
NXP's rt-sdk-ara2 configures ``/var/run/proxy.sock``, and the Kinara
runtime meta-kinara packages configures ``/var/run/ara2.sock``.
"""

def proxy_endpoint() -> ProxyEndpoint:
    """Resolve the endpoint :meth:`Session.connect` will open.

    Returns ``ARA2_SOCKET`` if it is set, otherwise the first entry in
    :data:`SOCKET_PATHS` that exists, otherwise a running proxy (UNIX
    before TCP). :meth:`Session.connect` resolves through this function and
    then makes a single connection attempt; this is exposed separately for
    callers that need to know where it will connect, such as the default
    value of a CLI flag.

    Raises:
        ProxyError: If no proxy endpoint can be found.
    """
    ...

# =============================================================================
# Exceptions
# =============================================================================

class Ara2Error(RuntimeError):
    """Base exception for all ARA-2 errors.

    Every instance has the attributes below. When the client library
    returned a failing status, ``code`` (the ``dv_status_code_t`` value) and
    ``name`` (the library's name for it) are set. On DVAPI 1.3 and later,
    calls with a detailed-status variant also set ``sub_code``,
    ``component`` (e.g. ``"proxy"``) and ``severity`` (e.g.
    ``"retryable"``). When the client library could not parse a model file
    (``DV_MODEL_LOAD_FAILURE``), ``compatibility`` says whether the model's
    format suits the runtime generation (``"mismatch"``, ``"matching"`` or
    ``"unknown"``) and ``model_format`` is the format read from the file
    (``None`` if it could not be read), and the message says what to do about it. Attributes that do not
    apply are ``None``.
    """

    code: int | None
    name: str | None
    sub_code: int | None
    component: str | None
    severity: str | None
    model_format: int | None
    compatibility: Literal["mismatch", "matching", "unknown"] | None

class LibraryError(Ara2Error):
    """Failed to load the ARA-2 client library.

    The library is loaded by name from NXP's ``imx-nxp-ara2`` package;
    several candidate names are tried because NXP's packaging has not
    settled on a stable soname.
    """

    ...

class HardwareError(Ara2Error):
    """NPU hardware fault or endpoint state error."""

    ...

class ProxyError(Ara2Error):
    """Proxy connection or communication error."""

    ...

class UnsupportedError(Ara2Error):
    """The loaded DVAPI version does not provide the requested feature.

    Raised by ``Endpoint.is_busy()`` and ``Session.set_log_level()`` on
    runtimes that lack them.
    """

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

class Abi:
    """The DVAPI generation a loaded ``libaraclient`` implements.

    This is the version of the ``dvapi.h`` interface, which governs struct
    layout -- not the version of the SDK the library was packaged in.
    """

    V1_1: Abi
    """DVAPI 1.1.x, shipped by the Kinara ARA-2 runtime meta-kinara packages."""
    V1_3: Abi
    """DVAPI 1.3.x, shipped by NXP's rt-sdk-ara2 as ``imx-nxp-ara2``."""
    V3_0: Abi
    """DVAPI 3.0.x, shipped by the arart r3.0 runtime."""

    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class State:
    """Endpoint operational state returned by ``Endpoint.check_status()``.

    The members are the union of all three DVAPI generations. DVAPI 1.3 and
    3.0 reuse values 4 and 5 for names 1.1 does not have, so ``ActiveBoosted``
    and ``ThermalInactive`` are only reported under 1.1, and
    ``ThermalActiveSlow`` and ``FailSafe`` only under 1.3 and 3.0.
    ``Session.abi`` reports which library is loaded.
    """

    Init: State
    """Device is initializing. Not yet ready for model loading."""
    Idle: State
    """Device is idle and ready to load models or run inference."""
    Active: State
    """Device is actively running inference at normal clock speed."""
    ActiveSlow: State
    """Device is running inference at reduced clock speed (power saving)."""
    ActiveBoosted: State
    """DVAPI 1.1 value 4, documented upstream as reduced clock speed despite
    its name. DVAPI 1.3 renamed the same value ``ThermalActiveSlow``."""
    ThermalInactive: State
    """DVAPI 1.1 value 5, thermally inactive. DVAPI 1.3 renamed it
    ``FailSafe`` without updating the upstream documentation."""
    ThermalActiveSlow: State
    """Device is running inference at reduced clock speed due to thermal
    limits. DVAPI 1.3 value 4."""
    FailSafe: State
    """Device is in a restricted fail-safe mode. DVAPI 1.3 value 5."""
    ThermalUnknown: State
    """Thermal state cannot be determined."""
    Inactive: State
    """Device is powered down or not available."""
    Fault: State
    """Unrecoverable hardware error. Restart the ARA-2 proxy service."""
    Unknown: State
    """A value neither DVAPI generation defines. The raw value is logged at
    warning level, since a Python enum member cannot carry it."""

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

class LogLevel:
    """Verbosity of the client library's own logging."""

    Trace: LogLevel
    Debug: LogLevel
    Info: LogLevel
    Warn: LogLevel
    Error: LogLevel
    Critical: LogLevel
    Off: LogLevel

    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ModelType:
    """The kind of model, as the runtime reports it.

    ``McsLlm`` is reported by DVAPI 3.0 only. ``Other`` is a value this
    build does not know for the loaded version; the first such raw value is
    logged at warning level."""

    Ara1Cnn: ModelType
    Ara2Cnn: ModelType
    Ara2Llm: ModelType
    Ara2LlmDynV2: ModelType
    McsLlm: ModelType
    Other: ModelType

    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class InterfaceKind:
    """The host bus an endpoint is attached through.

    ``Other`` is a host interface value this build does not recognise; the
    first such raw value is logged at warning level."""

    Pcie: InterfaceKind
    Usb: InterfaceKind
    Other: InterfaceKind

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
    largest_contiguous_free: int | None
    """Largest contiguous free block in bytes; ``None`` before DVAPI 3.0."""

class ModelTiming:
    """Timing information from a model inference run.

    Durations are float seconds, like ``time.monotonic()`` differences,
    and keep the runtime's sub-microsecond precision.
    """

    run_time: float
    """NPU inference execution time in seconds."""
    input_time: float
    """Input DMA transfer time in seconds."""
    output_time: float
    """Output DMA transfer time in seconds."""

class Clocks:
    """Clock frequencies in MHz."""

    sys_mhz: int
    """System core clock."""
    dram_mhz: int
    """DRAM clock."""
    nnp_mhz: int | None
    """Neural processor clock; ``None`` where the library does not report it."""
    sbp_mhz: int | None
    """SBP clock; ``None`` where the library does not report it."""

class InferenceStats:
    """Everything the runtime reports about one completed inference.

    Returned by :meth:`Model.run` and :meth:`InferRequest.wait`.

    The ``*_start`` timestamps are seconds on ``CLOCK_MONOTONIC``, the
    clock ``time.monotonic()`` reads. They are ``None`` on DVAPI 1.x,
    whose runtime stamps with the wall clock.
    """

    timing: ModelTiming
    """Execution and transfer durations."""
    clocks: Clocks
    """Clocks the inference ran at."""
    hw_cycles: int
    """Hardware cycles, including floating-point work."""
    fp_cycles: int
    """Hardware cycles spent on floating-point work."""
    input_start: float | None
    """When the input transfer started."""
    inference_start: float | None
    """When the inference entered the NNP queue."""
    output_start: float | None
    """When the output transfer started."""
    retries_session_total: int
    """Inference retries the proxy performed across this session."""
    retries_this_inference: int
    """Retries of this inference."""

class EndpointStatistics:
    """Live operational statistics for an endpoint."""

    state: State
    """Current operational state of the endpoint."""
    clocks: Clocks
    """Endpoint clocks."""
    core_voltage_v: float
    """Average core voltage across the measurement points, in volts.

    DVAPI 1.3 and later; the DVAPI 1.1 runtime (Kinara runtime 1.2.1) does
    not measure it and reports 0.0.
    """
    temp_c: float
    """Average endpoint temperature across the measurement points, in °C."""
    soft_reset_count: int
    """Soft resets the endpoint has performed."""
    dram: DramStatistics
    """DRAM usage statistics."""

class ChipInfo:
    """The ARA-2 chip an endpoint is built on."""

    id: str | None
    """Chip name, such as ``"ARA-2"``; ``None`` when the library reports none."""
    revision: str | None
    """Silicon revision; ``None`` when the library reports none."""
    control_processors: int
    """Number of control processors."""
    neural_processors: int
    """Number of neural processors."""
    l2_bytes: int
    """L2 memory size in bytes."""

class DramInfo:
    """The DRAM attached to an endpoint."""

    vendor_name: str | None
    """Manufacturer name, when the runtime knows it."""

class Interface:
    """Where an endpoint sits on the host."""

    kind: InterfaceKind
    """The host bus type."""
    bus: int
    """Bus number."""
    device: int
    """Device number."""
    port: int
    """Port number."""
    sysfs_path: os.PathLike[str] | None
    """The device's sysfs directory; reported for PCIe only."""

class EndpointInfo:
    """Static information about an endpoint's hardware."""

    device_id: int
    """PCI-style device identifier."""
    vendor_id: int
    """Vendor identifier."""
    chip: ChipInfo | None
    """The ARA-2 chip, when the runtime reports it."""
    dram: DramInfo | None
    """The attached DRAM, when the runtime reports it."""
    interface: Interface | None
    """How the endpoint is attached to the host, when the runtime reports it."""

class InputQuantization:
    """Input tensor quantization parameters.

    For qmode 9 models (the production default), ``qn`` is the per-tensor
    scale and ``offset`` is the integer zero-point. Per-channel image
    preprocessing (mean/scale, BGR swap, etc.) lives on
    :class:`InputPreprocess` instead.
    """

    qn: float
    """Per-tensor quantization scale."""
    offset: int
    """Integer zero-point offset."""
    is_signed: bool
    """True if the tensor uses signed int8, False for uint8."""
    qmode: int
    """Kinara quantization mode (9 = asymmetric, production default)."""

class InputPreprocess:
    """Image preprocessing parameters for an input tensor.

    Describes how float image data is expected to be normalized before
    quantization: ``(pixel - mean[c]) * scale[c]``. The per-channel
    ``mean`` and ``scale`` arrays are indexed by channel (C order after
    any BGR->RGB swap).
    """

    mean: tuple[float, float, float]
    """Per-channel mean subtracted during preprocessing."""
    scale: tuple[float, float, float]
    """Per-channel scale applied after mean subtraction."""
    bgr_to_rgb: bool
    """True if BGR inputs must be swapped to RGB before normalization."""
    aspect_resize: bool
    """True if the source image should be letterboxed to preserve aspect ratio."""
    mirror: bool
    """True if the input should be horizontally mirrored."""
    center_crop: bool
    """True if the input should be center-cropped to the model's input size."""

class OutputQuantization:
    """Output tensor quantization parameters (qmode 9 semantics).

    Use ``Model.dequantize()`` for automatic conversion, or apply
    manually: ``float_val = (raw - offset) * qn`` where ``raw`` is the
    signed or unsigned integer sample (per ``is_signed``).
    """

    qn: float
    """Per-tensor quantization scale. ``float_val = (raw - offset) * qn``."""
    offset: int
    """Integer zero-point offset."""
    is_signed: bool
    """True if the tensor uses signed int8, False for uint8."""

class InputTensorInfo:
    """Detailed information about an input tensor."""

    layer_id: int
    """Model layer this tensor belongs to."""
    blob_id: int
    """Blob index within the layer."""
    layer_name: str
    """Name of the layer, as compiled into the model."""
    blob_name: str
    """Name of the blob within the layer."""
    layer_type: str
    """Layer type reported by the compiler (e.g. "Convolution")."""
    layout: str
    """Data layout string (e.g., "NCHW")."""
    size: int
    """Total size in bytes."""
    width: int
    """Width in pixels."""
    height: int
    """Height in pixels."""
    nch: int
    """Number of channels."""
    bpp: int
    """Bytes per element."""
    batch_size: int
    """Number of batches this tensor holds."""
    quant: InputQuantization
    """Quantization parameters for converting to and from float."""
    preprocess: InputPreprocess
    """Image preprocessing parameters (per-channel mean/scale, BGR swap, etc.)."""

class OutputTensorInfo:
    """Detailed information about an output tensor."""

    layer_id: int
    """Model layer this tensor belongs to."""
    blob_id: int
    """Blob index within the layer."""
    fused_parent_id: int
    """Layer this one was fused into, or -1 when it was not fused."""
    layer_name: str
    """Name of the layer, as compiled into the model."""
    blob_name: str
    """Name of the blob within the layer."""
    layer_fused_parent_name: str
    """Name of the fused parent layer, empty when it was not fused."""
    layer_type: str
    """Layer type reported by the compiler (e.g. "Convolution")."""
    layout: str
    """Data layout string (e.g., "NCHW")."""
    size: int
    """Total size in bytes."""
    width: int
    """Width in pixels."""
    height: int
    """Height in pixels."""
    nch: int
    """Number of channels."""
    bpp: int
    """Bytes per element."""
    num_classes: int
    """Number of classes the model was trained on."""
    layer_output_type: ModelOutputType
    """What kind of output this layer produces."""
    max_dynamic_id: int
    """Highest batch id this tensor supports."""
    quant: OutputQuantization
    """Quantization parameters for converting to and from float."""

# =============================================================================
# Core Classes
# =============================================================================

class ProxyEndpoint:
    """An address a proxy accepts connections on."""

    @property
    def kind(self) -> Literal["unix", "tcp"]:
        """Transport: ``"unix"`` or ``"tcp"``."""
        ...

    @property
    def path(self) -> str | None:
        """Socket path, for a ``"unix"`` endpoint."""
        ...

    @property
    def host(self) -> str | None:
        """IPv4 address, for a ``"tcp"`` endpoint."""
        ...

    @property
    def port(self) -> int | None:
        """TCP port, for a ``"tcp"`` endpoint."""
        ...

    def connect(self) -> Session:
        """Open a session against this endpoint."""
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class Proxy:
    """A running ARA-2 proxy found by :func:`discover`."""

    @property
    def pid(self) -> int:
        """Process ID."""
        ...

    @property
    def endpoints(self) -> list[ProxyEndpoint]:
        """Where it accepts connections.

        Empty when neither reading the process's file descriptors nor its
        configuration file was possible.
        """
        ...

    @property
    def config(self) -> str | None:
        """Configuration file named on its command line, if it named one."""
        ...

    @property
    def exe(self) -> str | None:
        """Its executable, if readable."""
        ...

    def connect(self) -> Session:
        """Open a session against this proxy's first endpoint."""
        ...

    def __repr__(self) -> str: ...

def discover() -> list[Proxy]:
    """Find every ARA-2 proxy currently running on this host.

    Returns a list because more than one proxy can be running; an empty
    list means none was found, which is not an error.

    Endpoints are resolved by observing the sockets each process holds
    open, which reflects where it is really listening however that was
    decided. That needs permission to read the process's file descriptors
    -- the proxy runs as root under both packagings -- so when it is
    unavailable this falls back to parsing the configuration file named
    on the command line.

    Example:
        >>> for proxy in edgefirst_ara2.discover():
        ...     print(proxy.pid, [str(e) for e in proxy.endpoints])
    """
    ...

class ComponentVersion:
    """A component and the version the proxy reports for it."""

    @property
    def component(self) -> str:
        """Component name, e.g. ``"proxy"``, ``"firmware"``, ``"client_lib"``.

        ``"product_type_N"`` for a component this build does not name.
        """
        ...

    @property
    def version(self) -> str:
        """Version as ``"major.minor.patch.patch_minor"``."""
        ...

class Session:
    """ARA-2 session for communicating with the proxy.

    A Session represents a connection to the ARA-2 proxy service.
    Supports the context manager protocol for resource management::

        with Session.connect() as session:
            endpoints = session.list_endpoints()

    Example::

        session = Session.connect()
        versions = session.versions()
        endpoints = session.list_endpoints()
        session.close()
    """

    @staticmethod
    def connect() -> Session:
        """Create a session connected to the proxy.

        Connects to the endpoint :func:`proxy_endpoint` resolves:
        ``ARA2_SOCKET`` if set, otherwise the first entry in
        :data:`SOCKET_PATHS` that exists, otherwise a running proxy. This
        is the recommended way to connect; use
        :meth:`create_via_unix_socket` when the path must be hardcoded or
        is chosen some other way.

        Raises:
            ProxyError: If no proxy endpoint is found, or the connection
                to the endpoint fails.
        """
        ...

    @staticmethod
    def from_config(path: str | os.PathLike[str]) -> Session:
        """Create a session from a proxy configuration file.

        Reads the ``proxy:`` section's ``interface_type`` and the address
        list it selects, then connects to the first endpoint declared --
        a UNIX socket for ``SOCKET``, a TCP/IPv4 address for ``IPV4``.
        Use this when the configuration is known but the socket path is
        not, which is the usual case across packagings: NXP's rt-sdk-ara2
        reads ``/etc/rt-sdk-ara240/proxy_config.yaml`` and the Kinara
        runtime meta-kinara packages reads ``/etc/ara2.yaml``, and the two
        name different sockets.

        Args:
            path: Path to the proxy YAML (str or os.PathLike).

        Raises:
            ProxyError: If the file cannot be read or understood, or it
                declares no reachable endpoint.
        """
        ...

    @staticmethod
    def create_via_unix_socket(socket_path: str | os.PathLike[str]) -> Session:
        """Create a session connected via UNIX domain socket.

        Args:
            socket_path: Path to the UNIX socket (e.g., "/var/run/proxy.sock")

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

    def versions(self) -> list[ComponentVersion]:
        """Versions of the components currently running.

        Returns:
            One :class:`ComponentVersion` per running component.
        """
        ...

    def supported_versions(self) -> list[ComponentVersion]:
        """Every component version the proxy supports.

        May list a component more than once.
        """
        ...

    @property
    def supports_fd_passing(self) -> bool:
        """Whether the session can use shared-memory and DMA-BUF tensors.

        True for UNIX-socket sessions; False over TCP, where tensors live
        in heap memory with no file descriptor, so the tensor fd accessors
        raise ``TensorError`` and data is copied instead.
        """
        ...

    def list_endpoints(self) -> list[Endpoint]:
        """List all available ARA-2 endpoints."""
        ...

    def inflight_count(self) -> int:
        """Get the number of in-flight inference requests.

        Returns the count of requests submitted via :meth:`Model.submit`
        that the client library has not yet received a response for.
        Useful for monitoring pipeline depth.
        """
        ...

    @property
    def socket_type(self) -> Literal["unix", "tcp"]:
        """Socket type used for this connection."""
        ...

    @property
    def abi(self) -> Abi:
        """DVAPI generation of the ``libaraclient`` this session uses.

        Probed when the library was opened. It determines which of the
        overlapping :class:`State` members can be reported.
        """
        ...

    @property
    def dvapi_version(self) -> str | None:
        """DVAPI version the client library reported, e.g. ``"1.3.2.0"``.

        ``None`` if the library would not report one. This is the
        interface version, not the version of the SDK it was packaged in.
        """
        ...

    def set_log_level(self, level: LogLevel) -> None:
        """Set the client library's log verbosity. DVAPI 1.3 and later.

        The level belongs to the client library, not the session, so it
        applies to every session in the process.

        Raises:
            UnsupportedError: If the loaded library does not support it.
        """
        ...

    def close(self) -> None:
        """Close this Python session handle.

        After calling ``close()``, any further method call on this
        Session raises ``Ara2Error``. This only drops this Python
        handle; the underlying proxy connection may remain open while
        other objects created from this session (such as ``Endpoint``
        or ``Model``) are still alive. The underlying connection is
        released when the last handle referencing it is dropped. Safe
        to call multiple times (idempotent).
        """
        ...

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool: ...
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

    def statistics(self) -> EndpointStatistics:
        """Live statistics: state, clocks, voltage, temperature, soft
        resets and DRAM."""
        ...

    def info(self) -> EndpointInfo:
        """Static information about the endpoint's hardware."""
        ...

    def is_busy(self) -> bool:
        """Whether an inference is running or queued on the endpoint now.

        Raises:
            UnsupportedError: If the loaded runtime lacks the call.
        """
        ...

    def load_model(self, model_path: str | os.PathLike[str]) -> Model:
        """Load a neural network model from a .dvm file.

        Args:
            model_path: Path to the compiled model file (.dvm)

        The file's model format is read first, and a model built for the
        other runtime generation is logged as a warning before loading.

        Raises:
            ModelError: If the client library could not parse the file
                (code 520, ``DV_MODEL_LOAD_FAILURE``), with
                ``compatibility`` and ``model_format`` set and a message
                saying whether to reconvert the model.
            HardwareError: If the endpoint has insufficient DRAM.
            Ara2Error: Or one of its subclasses, for any other failure.
        """
        ...

    def __repr__(self) -> str: ...

class Model:
    """Neural network model loaded on an endpoint.

    Supports the context manager protocol. The model is unloaded from the
    NPU when the last Python reference is garbage-collected (via Rust Drop).
    Using ``with`` ensures references are released promptly on scope exit::

        with endpoint.load_model("model.dvm") as model:
            model.allocate_tensors("dma")
            model.set_input_tensor(0, input_data)
            stats = model.run()
            output = model.get_output_tensor(0)

    For zero-copy preprocessing with edgefirst-image::

        import edgefirst.image as ef_image

        processor = ef_image.ImageProcessor()
        fd = model.input_tensor_fd(0)
        dst = processor.import_image(fd, w, h, ef_image.PixelFormat.PlanarRgb)
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
                    ``None`` (auto-select, tries DMA first; heap on a TCP
                    session, which cannot pass file descriptors). Use
                    ``"dma"`` for zero-copy workflows with edgefirst-image.

        Over TCP the tensors are heap memory without a file descriptor, so
        ``input_tensor_fd()`` and ``output_tensor_fd()`` raise
        ``TensorError``. Check ``Session.supports_fd_passing`` and copy
        data with ``set_input_tensor()`` and ``get_output_tensor()`` instead.
        """
        ...

    def set_timeout(self, timeout: float) -> None:
        """Set the ``run()`` timeout in seconds (default: 1.0).

        The timeout is rounded up to whole milliseconds.

        Raises:
            ValueError: If ``timeout`` is negative, not finite, or too large.
        """
        ...

    def run(self) -> InferenceStats:
        """Run inference on the model.

        Returns:
            InferenceStats: Timing, clocks, cycle counts and timestamps.

        Raises:
            TensorError: If tensors have not been allocated.
            ModelError: If inference times out (increase with ``set_timeout``).
            HardwareError: If the NPU is in a fault or thermal state.
        """
        ...

    def submit(self) -> InferRequest:
        """Submit inference asynchronously.

        Returns an :class:`InferRequest` handle immediately without
        blocking. Call :meth:`InferRequest.wait` to retrieve the result
        once the NPU finishes.

        This enables overlapping CPU work (e.g., preprocessing the next
        frame) with NPU execution — the key building block for pipeline
        parallelism.

        The model's tensors must not be reallocated (via
        ``allocate_tensors()``) while the returned request is pending.

        Example::

            request = model.submit()
            # ... preprocess next frame while NPU is busy ...
            timing = request.wait()

        Raises:
            TensorError: If tensors have not been allocated.
        """
        ...

    def close(self) -> None:
        """Unload the model and release its resources.

        After calling ``close()``, any further method call raises
        ``Ara2Error``. Safe to call multiple times (idempotent).
        """
        ...

    # -- Tensor I/O (numpy) --

    def set_input_tensor(self, index: int, data: npt.NDArray[Any]) -> None:
        """Copy numpy array data into an input tensor.

        The input is dtype-agnostic: any numpy dtype (``uint8``, ``int8``,
        ``uint16``, ``int16``, ``float32``, etc.) is accepted. Internally
        the array is flattened to bytes via ``tobytes()`` and copied
        verbatim into the tensor memory, so the buffer layout must already
        match what the model expects — this method performs no dtype
        conversion or reshape.

        The total byte count (``data.nbytes``) must equal
        ``input_size(index)``; non-contiguous arrays are made contiguous
        by ``tobytes()`` before the copy.

        Args:
            index: Input tensor index (0-based)
            data: numpy array whose raw byte buffer matches the tensor
                  size. Dtype is not checked beyond its contribution to
                  ``nbytes``.

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or sizes don't match.
        """
        ...

    def get_output_tensor(self, index: int) -> npt.NDArray[Any]:
        """Get output tensor data as a typed, shaped numpy array.

        The numpy dtype is derived from the tensor's quantization
        metadata ``(bpp, is_signed)``:

        - ``bpp == 1`` and not signed → ``uint8``
        - ``bpp == 1`` and ``is_signed`` → ``int8``
        - ``bpp == 2`` and not signed → ``uint16``
        - ``bpp == 2`` and ``is_signed`` → ``int16``
        - ``bpp == 4`` → ``float32``

        The returned array is reshaped to the tensor's declared shape
        ``(channels, height, width)``; callers that need the legacy flat
        layout can call ``.ravel()``.

        Args:
            index: Output tensor index (0-based)

        Returns:
            numpy.ndarray: Shaped, typed view of the output tensor data.

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or the tensor has
                an unsupported bpp.
        """
        ...

    def dequantize(self, index: int) -> npt.NDArray[np.float32]:
        """Dequantize an output tensor to ``float32``.

        Applies ``float_val = (raw - offset) * qn`` using the output
        tensor's ``(qn, offset)`` pair, where ``raw`` is the signed or
        unsigned integer sample. Currently only qmode 9 (asymmetric) is
        supported; the model's qmode is read from input 0 and other modes
        raise ``Ara2Error``.

        The returned array has the tensor's declared ``(C, H, W)`` shape.

        Args:
            index: Output tensor index (0-based)

        Returns:
            numpy.ndarray: ``float32`` array reshaped to ``(C, H, W)``.

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or the tensor has
                an unsupported bpp.
            Ara2Error: If the model uses an unsupported qmode.
        """
        ...

    # -- DMA-BUF zero-copy --

    def input_tensor_fd(self, index: int) -> int:
        """Get a cloned DMA-BUF file descriptor for an input tensor.

        The returned FD is owned by the caller. Pass it to
        ``edgefirst.image.ImageProcessor.import_image()`` for zero-copy GPU
        preprocessing. Close with ``os.close(fd)`` when done.

        Args:
            index: Input tensor index (0-based)

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or use system memory
                (always the case over TCP).
        """
        ...

    def output_tensor_fd(self, index: int) -> int:
        """Get a cloned DMA-BUF file descriptor for an output tensor.

        The returned FD is owned by the caller. Pass it to
        ``edgefirst.tensor.Tensor.from_fd()`` to decode the raw NPU output
        without a copy — that call takes ownership of the FD. Otherwise
        close it with ``os.close(fd)`` when done.

        Args:
            index: Output tensor index (0-based)

        Raises:
            IndexError: If index is out of range.
            TensorError: If tensors are not allocated or use system memory
                (always the case over TCP).
        """
        ...

    def input_tensor_memory(self, index: int) -> Literal["dma", "shm", "mem"]:
        """Get the memory type of an input tensor."""
        ...

    def output_tensor_memory(self, index: int) -> Literal["dma", "shm", "mem"]:
        """Get the memory type of an output tensor."""
        ...

    # -- Introspection --

    @property
    def version(self) -> str:
        """Model format version the model was compiled to, as
        ``"major.minor.patch.patch_minor"``: 2.0.0.0 for DVAPI 1.x runtimes,
        3.0.0.0 for 3.x. Its major is what ``read_model_format()`` returns."""
        ...

    @property
    def name(self) -> str | None:
        """Name given when the model was loaded, if any."""
        ...

    @property
    def model_type(self) -> ModelType:
        """The kind of model the runtime reports."""
        ...

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

    def input_preprocess(self, index: int) -> InputPreprocess:
        """Get image preprocessing parameters for an input tensor.

        Returns the per-channel ``mean`` and ``scale`` used to normalize
        float input data before quantization, plus layout-affecting flags
        (``bgr_to_rgb``, ``aspect_resize``, ``mirror``, ``center_crop``).

        Raises:
            IndexError: If index is out of range.
        """
        ...

    def output_quants(self, index: int) -> OutputQuantization:
        """Get quantization parameters for an output tensor."""
        ...

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool: ...
    def __repr__(self) -> str: ...

class InferRequest:
    """A pending asynchronous inference request.

    Created by :meth:`Model.submit`. Call :meth:`wait` to block until
    the NPU finishes and retrieve its statistics. The GIL is released
    during ``wait()`` so other Python threads can run concurrently.

    The underlying request is freed automatically when the object is
    garbage-collected or when ``wait()`` returns. Dropping or
    garbage-collecting a request that was never waited on blocks while it
    finishes, for up to about 4 s.

    Example::

        request = model.submit()
        # ... preprocess next frame while NPU is busy ...
        stats = request.wait()
        print(f"NPU inference: {stats.timing.run_time * 1e3:.2f} ms")

    Warning:
        Do not call ``model.allocate_tensors()`` while an InferRequest
        is still pending — the NPU is reading/writing the tensor buffers.
    """

    def wait(self, *, timeout: float = 1.0) -> InferenceStats:
        """Block until the inference completes and return its statistics.

        The GIL is released during the wait, allowing other Python
        threads to run concurrently.

        Consumes the request — calling ``wait()`` a second time raises
        ``Ara2Error``.

        Args:
            timeout: Maximum wait time in seconds (default: 1.0), rounded
                up to whole milliseconds. Increase for large models.

        Returns:
            InferenceStats: Timing, clocks, cycle counts and timestamps for
            the completed inference.

        Raises:
            ValueError: If ``timeout`` is negative, not finite, or too large.
            Ara2Error: If the request was already consumed, inference
                       failed, or the wait timed out.
        """
        ...

    @property
    def request_id(self) -> int:
        """Proxy-assigned request ID for log correlation.

        Each submitted request gets a unique ID from the proxy. Use it
        to correlate client events with ``journalctl -u ara2`` logs.

        Raises:
            Ara2Error: If the request has already been consumed.
        """
        ...

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

def read_model_format(path: str | os.PathLike[str]) -> int | None:
    """Read the model format a DVM file was compiled to.

    Converters built on the DVAPI 1.x SDKs write format 2 and those built
    on the DVAPI 3.0 SDK write format 3. Format 2 models
    load on DVAPI 1.x runtimes only, format 3 on DVAPI 3.x only. Only the
    parts of the file that hold the format are read.

    Returns:
        The format, or ``None`` when the file's structure is not
        recognised (it is not then guessed at).

    Raises:
        MetadataError: If the file cannot be read.
    """
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
    def ara2(self) -> Ara2Info | None:
        """Kinara ARA-2 specific metadata section (quantization mode, etc.)."""
        ...
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
    def model_name(self) -> str | None:
        """Model name for deployment."""
        ...
    @property
    def name(self) -> str | None:
        """Human-readable deployment name."""
        ...
    @property
    def author(self) -> str | None:
        """Author or organization."""
        ...
    @property
    def description(self) -> str | None:
        """Description of the deployed model."""
        ...

class InputSpec:
    """Input specification from model metadata."""

    @property
    def size(self) -> str | None:
        """Size string (e.g., "640x480")."""
        ...
    @property
    def input_channels(self) -> int | None:
        """Number of input channels."""
        ...
    @property
    def output_channels(self) -> int | None:
        """Number of output channels."""
        ...
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
    def index(self) -> int | None:
        """Output index."""
        ...
    @property
    def name(self) -> str | None:
        """Output name."""
        ...
    @property
    def output_type(self) -> str | None:
        """Output type (detection, segmentation, etc.)."""
        ...
    @property
    def decoder(self) -> str | None:
        """Decoder to use."""
        ...
    @property
    def decode(self) -> bool:
        """Whether to decode this output."""
        ...
    @property
    def dtype(self) -> str | None:
        """Data type (float32, int8, etc.)."""
        ...
    @property
    def shape(self) -> list[int]:
        """Tensor shape."""
        ...
    @property
    def dshape(self) -> list[tuple[str, int]]:
        """Physical axis names in memory order, as ``(name, extent)`` pairs.

        Names use the metadata spellings (``"batch"``, ``"num_boxes"``,
        ``"num_protos"``, ...), which map onto ``edgefirst.decoder.DimName``.
        Empty when the producer omitted the field.
        """
        ...
    @property
    def normalized(self) -> bool | None:
        """Whether box coordinates are already normalized to ``[0, 1]``.

        ``None`` when the metadata does not say — the caller must then treat
        the coordinates as pixel-space and divide the box quantization scale
        by the model's input dimension. Only meaningful for ``boxes`` and
        ``detection`` outputs.
        """
        ...

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
    def ppa(self) -> PpaMetrics | None:
        """Performance, power and area metrics."""
        ...

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

class Ara2Info:
    """Kinara-specific metadata from the ``ara2`` section of ``edgefirst.json``.

    Exposes ARA-2 / Kinara-specific fields carried in the DVM's
    EdgeFirst metadata: the quantization mode and the conversion
    provenance. Every field is ``None`` when the metadata does not carry it.
    """

    @property
    def qmode(self) -> int | None:
        """Kinara quantization mode (9 = asymmetric, the production default).

        Returns ``None`` if the DVM's metadata does not carry a
        ``qmode`` field in its ``ara2`` section.
        """
        ...
    @property
    def converter_version(self) -> str | None:
        """Version of the converter that produced the model.

        Read from ``converter_version``, or from ``version`` where a
        converter wrote that instead.
        """
        ...
    @property
    def sdk_version(self) -> str | None:
        """Version of the Kinara SDK the converter was built on."""
        ...
    @property
    def dvm_format(self) -> int | None:
        """Model format the converter recorded (2 for DVAPI 1.x runtimes, 3 for 3.x)."""
        ...

__all__ = [
    "__version__",
    # Constants
    "DEFAULT_SOCKET",
    "SOCKET_PATHS",
    # Functions
    "proxy_endpoint",
    "discover",
    # Exceptions
    "Ara2Error",
    "LibraryError",
    "HardwareError",
    "ProxyError",
    "UnsupportedError",
    "ModelError",
    "TensorError",
    "MetadataError",
    # Enums
    "Abi",
    "State",
    "ModelOutputType",
    "LogLevel",
    "ModelType",
    "InterfaceKind",
    # Data classes
    "ComponentVersion",
    "DramStatistics",
    "ModelTiming",
    "Clocks",
    "InferenceStats",
    "EndpointStatistics",
    "ChipInfo",
    "DramInfo",
    "Interface",
    "EndpointInfo",
    "InputQuantization",
    "InputPreprocess",
    "OutputQuantization",
    "InputTensorInfo",
    "OutputTensorInfo",
    # Core classes
    "Session",
    "Proxy",
    "ProxyEndpoint",
    "Endpoint",
    "Model",
    "InferRequest",
    # Metadata
    "read_metadata",
    "read_labels",
    "has_metadata",
    "read_model_format",
    "DvmMetadata",
    "Ara2Info",
    "DatasetInfo",
    "ModelInfo",
    "DeploymentInfo",
    "InputSpec",
    "OutputSpec",
    "CompilationInfo",
    "PpaMetrics",
]
