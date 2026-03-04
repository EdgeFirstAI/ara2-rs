"""
EdgeFirst ARA-2 Python Library Type Stubs

This module provides type hints for the edgefirst_ara2 library.
The actual implementation is in Rust via PyO3.
"""

from typing import Optional
from enum import Enum

class State(Enum):
    """Endpoint state enumeration"""
    Init: State
    Idle: State
    Active: State
    ActiveSlow: State
    ActiveBoosted: State
    ThermalInactive: State
    ThermalUnknown: State
    Inactive: State
    Fault: State

class OutputQuantization(Enum):
    """Output quantization type"""
    None_: OutputQuantization
    U8: OutputQuantization
    I8: OutputQuantization

class DramStatistics:
    """DRAM statistics for an endpoint"""
    @property
    def dram_size(self) -> int:
        """Total DRAM size in bytes"""
        ...

    @property
    def dram_occupancy_size(self) -> int:
        """DRAM occupancy size in bytes"""
        ...

    @property
    def free_size(self) -> int:
        """Free DRAM size in bytes"""
        ...

    @property
    def reserved_occupancy_size(self) -> int:
        """Reserved occupancy size in bytes"""
        ...

    @property
    def model_occupancy_size(self) -> int:
        """Model occupancy size in bytes"""
        ...

    @property
    def tensor_occupancy_size(self) -> int:
        """Tensor occupancy size in bytes"""
        ...

class ModelTiming:
    """Model timing information"""
    @property
    def run_time_us(self) -> int:
        """Model run time in microseconds"""
        ...

    @property
    def input_time_us(self) -> int:
        """Input transfer time in microseconds"""
        ...

    @property
    def output_time_us(self) -> int:
        """Output transfer time in microseconds"""
        ...

class Model:
    """Neural network model loaded on an endpoint"""

    def run(self) -> ModelTiming:
        """
        Run inference on the model

        Returns:
            ModelTiming: Timing information for the inference run
        """
        ...

    def n_inputs(self) -> int:
        """Get the number of input tensors"""
        ...

    def n_outputs(self) -> int:
        """Get the number of output tensors"""
        ...

class Endpoint:
    """ARA-2 accelerator endpoint"""

    def check_status(self) -> State:
        """
        Check the current status/state of the endpoint

        Returns:
            State: Current endpoint state
        """
        ...

    def dram_statistics(self) -> DramStatistics:
        """
        Get DRAM statistics for the endpoint

        Returns:
            DramStatistics: Memory usage information
        """
        ...

    def load_model(
        self,
        model_path: str,
        output_quantization: Optional[OutputQuantization] = None
    ) -> Model:
        """
        Load a neural network model from a file

        Args:
            model_path: Path to the model file
            output_quantization: Optional output quantization type

        Returns:
            Model: Loaded model ready for inference
        """
        ...

class Session:
    """ARA-2 session for communicating with the proxy"""

    @staticmethod
    def create_via_unix_socket(socket_path: str) -> Session:
        """
        Create a session connected via UNIX domain socket

        Args:
            socket_path: Path to the UNIX socket (e.g., "/var/run/ara2.sock")

        Returns:
            Session: A new session connected to the proxy
        """
        ...

    @staticmethod
    def create_via_tcp_ipv4_socket(ip: str, port: int) -> Session:
        """
        Create a session connected via TCP/IPv4 socket

        Args:
            ip: IPv4 address as a string (e.g., "127.0.0.1")
            port: Port number

        Returns:
            Session: A new session connected to the proxy
        """
        ...

    def versions(self) -> dict[str, str]:
        """
        Get version information for all components

        Returns:
            Dictionary mapping component names to version strings
        """
        ...

    def list_endpoints(self) -> list[Endpoint]:
        """
        List all available endpoints

        Returns:
            List of available ARA-2 endpoints
        """
        ...

def version() -> str:
    """
    Get the version of the edgefirst_ara2 Python library

    Returns:
        Version string
    """
    ...

__all__ = [
    "Session",
    "Endpoint",
    "Model",
    "State",
    "DramStatistics",
    "ModelTiming",
    "OutputQuantization",
    "version",
]
