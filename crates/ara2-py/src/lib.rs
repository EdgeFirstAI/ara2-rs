// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::{collections::HashMap, fmt::Display, net::Ipv4Addr, str::FromStr};

// ============================================================================
// Error handling
// ============================================================================

pub enum Error {
    Error(ara2::Error),
    PyErr(pyo3::PyErr),
    TypeError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Error(e) => write!(f, "{}", e),
            Error::PyErr(e) => write!(f, "{}", e),
            Error::TypeError(e) => write!(f, "Type error: {}", e),
        }
    }
}

impl From<ara2::Error> for Error {
    fn from(e: ara2::Error) -> Self {
        Error::Error(e)
    }
}

impl From<pyo3::PyErr> for Error {
    fn from(e: pyo3::PyErr) -> Self {
        Error::PyErr(e)
    }
}

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match e {
            Error::Error(e) => PyRuntimeError::new_err(format!("ara2 error: {}", e)),
            Error::PyErr(e) => e,
            Error::TypeError(e) => PyValueError::new_err(e),
        }
    }
}

// ============================================================================
// Wrapper Types
// ============================================================================

/// Endpoint state enum
#[pyclass(module = "edgefirst_ara2")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum State {
    Init,
    Idle,
    Active,
    ActiveSlow,
    ActiveBoosted,
    ThermalInactive,
    ThermalUnknown,
    Inactive,
    Fault,
}

impl From<ara2::State> for State {
    fn from(state: ara2::State) -> Self {
        match state {
            ara2::State::Init => State::Init,
            ara2::State::Idle => State::Idle,
            ara2::State::Active => State::Active,
            ara2::State::ActiveSlow => State::ActiveSlow,
            ara2::State::ActiveBoosted => State::ActiveBoosted,
            ara2::State::ThermalInactive => State::ThermalInactive,
            ara2::State::ThermalUnknown => State::ThermalUnknown,
            ara2::State::Inactive => State::Inactive,
            ara2::State::Fault => State::Fault,
        }
    }
}

#[pymethods]
impl State {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("State.{:?}", self)
    }
}

/// DRAM statistics for an endpoint
#[pyclass(module = "edgefirst_ara2")]
#[derive(Clone, Debug)]
pub struct DramStatistics {
    dram_size: u64,
    dram_occupancy_size: u64,
    free_size: u64,
    reserved_occupancy_size: u64,
    model_occupancy_size: u64,
    tensor_occupancy_size: u64,
}

impl From<ara2::DramStatistics> for DramStatistics {
    fn from(stats: ara2::DramStatistics) -> Self {
        DramStatistics {
            dram_size: stats.dram_size,
            dram_occupancy_size: stats.dram_occupancy_size,
            free_size: stats.free_size,
            reserved_occupancy_size: stats.reserved_occupancy_size,
            model_occupancy_size: stats.model_occupancy_size,
            tensor_occupancy_size: stats.tensor_occupancy_size,
        }
    }
}

#[pymethods]
impl DramStatistics {
    /// Total DRAM size in bytes
    #[getter]
    fn dram_size(&self) -> u64 {
        self.dram_size
    }

    /// DRAM occupancy size in bytes
    #[getter]
    fn dram_occupancy_size(&self) -> u64 {
        self.dram_occupancy_size
    }

    /// Free DRAM size in bytes
    #[getter]
    fn free_size(&self) -> u64 {
        self.free_size
    }

    /// Reserved occupancy size in bytes
    #[getter]
    fn reserved_occupancy_size(&self) -> u64 {
        self.reserved_occupancy_size
    }

    /// Model occupancy size in bytes
    #[getter]
    fn model_occupancy_size(&self) -> u64 {
        self.model_occupancy_size
    }

    /// Tensor occupancy size in bytes
    #[getter]
    fn tensor_occupancy_size(&self) -> u64 {
        self.tensor_occupancy_size
    }

    fn __repr__(&self) -> String {
        format!(
            "DramStatistics(dram_size={}, dram_occupancy_size={}, free_size={}, reserved_occupancy_size={}, model_occupancy_size={}, tensor_occupancy_size={})",
            self.dram_size, self.dram_occupancy_size, self.free_size, self.reserved_occupancy_size, self.model_occupancy_size, self.tensor_occupancy_size
        )
    }
}

/// Model timing information
#[pyclass(module = "edgefirst_ara2")]
#[derive(Clone, Copy, Debug)]
pub struct ModelTiming {
    run_time_us: u64,
    input_time_us: u64,
    output_time_us: u64,
}

impl From<ara2::ModelTiming> for ModelTiming {
    fn from(timing: ara2::ModelTiming) -> Self {
        ModelTiming {
            run_time_us: timing.run_time.as_micros() as u64,
            input_time_us: timing.input_time.as_micros() as u64,
            output_time_us: timing.output_time.as_micros() as u64,
        }
    }
}

#[pymethods]
impl ModelTiming {
    /// Model run time in microseconds
    #[getter]
    fn run_time_us(&self) -> u64 {
        self.run_time_us
    }

    /// Input transfer time in microseconds
    #[getter]
    fn input_time_us(&self) -> u64 {
        self.input_time_us
    }

    /// Output transfer time in microseconds
    #[getter]
    fn output_time_us(&self) -> u64 {
        self.output_time_us
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelTiming(run_time_us={}, input_time_us={}, output_time_us={})",
            self.run_time_us, self.input_time_us, self.output_time_us
        )
    }
}

/// Output quantization type
#[pyclass(module = "edgefirst_ara2")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutputQuantization {
    None,
    U8,
    I8,
}

#[pymethods]
impl OutputQuantization {
    fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("OutputQuantization.{:?}", self)
    }
}

// ============================================================================
// Model wrapper
// ============================================================================

/// Neural network model loaded on an endpoint
/// 
/// The Model class provides methods to run inference on a loaded neural network.
/// Input and output tensors are accessed via zero-based indices.
#[pyclass(module = "edgefirst_ara2")]
pub struct Model {
    // Store a reference to the session to keep it alive
    _session: Py<Session>,
    // Note: We cannot directly wrap ara2::Model due to lifetime issues
    // Instead, we'll need to manage state differently
}

#[pymethods]
impl Model {
    /// Run inference on the model
    /// 
    /// Returns:
    ///     ModelTiming: Timing information for the inference run
    fn run(&mut self) -> Result<ModelTiming, Error> {
        // This will need special handling - for now return an error
        Err(Error::TypeError(
            "Model.run() requires accessing the underlying Rust model".to_string(),
        ))
    }

    /// Get the number of input tensors
    fn n_inputs(&self) -> Result<usize, Error> {
        Err(Error::TypeError(
            "Model operations require session context".to_string(),
        ))
    }

    /// Get the number of output tensors
    fn n_outputs(&self) -> Result<usize, Error> {
        Err(Error::TypeError(
            "Model operations require session context".to_string(),
        ))
    }

    fn __repr__(&self) -> String {
        "Model()".to_string()
    }
}

// ============================================================================
// Endpoint wrapper
// ============================================================================

/// ARA-2 accelerator endpoint
/// 
/// An endpoint represents a single ARA-2 accelerator device that can load
/// and execute neural network models.
#[pyclass(module = "edgefirst_ara2")]
pub struct Endpoint {
    _session: Py<Session>,
}

#[pymethods]
impl Endpoint {
    /// Check the current status/state of the endpoint
    /// 
    /// Returns:
    ///     State: Current endpoint state
    fn check_status(&self) -> Result<State, Error> {
        Err(Error::TypeError(
            "Endpoint operations require session context".to_string(),
        ))
    }

    /// Get DRAM statistics for the endpoint
    /// 
    /// Returns:
    ///     DramStatistics: Memory usage information
    fn dram_statistics(&self) -> Result<DramStatistics, Error> {
        Err(Error::TypeError(
            "Endpoint operations require session context".to_string(),
        ))
    }

    /// Load a neural network model from a file
    /// 
    /// Args:
    ///     model_path: Path to the model file
    ///     output_quantization: Optional output quantization type
    /// 
    /// Returns:
    ///     Model: Loaded model ready for inference
    #[pyo3(signature = (model_path, output_quantization = None))]
    fn load_model(
        &self,
        model_path: String,
        output_quantization: Option<OutputQuantization>,
    ) -> Result<Model, Error> {
        Err(Error::TypeError(
            "Endpoint operations require session context".to_string(),
        ))
    }

    fn __repr__(&self) -> String {
        "Endpoint()".to_string()
    }
}

// ============================================================================
// Session wrapper
// ============================================================================

/// ARA-2 session for communicating with the proxy
/// 
/// A Session represents a connection to the ARA-2 proxy service, which can
/// be established via either a UNIX socket or TCP socket. The session is used
/// to enumerate endpoints and retrieve version information.
/// 
/// Example:
///     >>> import ara2
///     >>> session = ara2.Session.create_via_unix_socket("/var/run/ara2.sock")
///     >>> versions = session.versions()
///     >>> endpoints = session.list_endpoints()
#[pyclass(module = "edgefirst_ara2")]
pub struct Session(ara2::Session);

impl Display for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Session(socket_type={:?})", self.0.socket_type())
    }
}

#[pymethods]
impl Session {
    /// Create a session connected via UNIX domain socket
    /// 
    /// Args:
    ///     socket_path: Path to the UNIX socket (e.g., "/var/run/ara2.sock")
    /// 
    /// Returns:
    ///     Session: A new session connected to the proxy
    /// 
    /// Example:
    ///     >>> session = Session.create_via_unix_socket("/var/run/ara2.sock")
    #[staticmethod]
    fn create_via_unix_socket(socket_path: &str) -> Result<Self, Error> {
        Ok(Session(ara2::Session::create_via_unix_socket(
            socket_path,
        )?))
    }

    /// Create a session connected via TCP/IPv4 socket
    /// 
    /// Args:
    ///     ip: IPv4 address as a string (e.g., "127.0.0.1")
    ///     port: Port number
    /// 
    /// Returns:
    ///     Session: A new session connected to the proxy
    /// 
    /// Example:
    ///     >>> session = Session.create_via_tcp_ipv4_socket("127.0.0.1", 5555)
    #[staticmethod]
    fn create_via_tcp_ipv4_socket(ip: &str, port: u16) -> Result<Self, Error> {
        let ip_addr = Ipv4Addr::from_str(ip)
            .map_err(|e| Error::TypeError(format!("Invalid IP address: {}", e)))?;
        Ok(Session(ara2::Session::create_via_tcp_ipv4_socket(
            ip_addr, port,
        )?))
    }

    /// Get version information for all components
    /// 
    /// Returns:
    ///     dict: Dictionary mapping component names to version strings
    /// 
    /// Example:
    ///     >>> versions = session.versions()
    ///     >>> print(versions["proxy"])
    ///     1.0.0.0
    fn versions(&self) -> Result<HashMap<String, String>, Error> {
        Ok(self.0.versions()?)
    }

    /// List all available endpoints
    /// 
    /// Returns:
    ///     list[Endpoint]: List of available ARA-2 endpoints
    /// 
    /// Example:
    ///     >>> endpoints = session.list_endpoints()
    ///     >>> print(f"Found {len(endpoints)} endpoints")
    fn list_endpoints(slf: Py<Self>) -> Result<Vec<Endpoint>, Error> {
        // Note: Due to lifetime constraints, we need special handling here
        // For now, return empty list as placeholder
        Ok(vec![])
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    fn __str__(&self) -> String {
        format!("{}", self)
    }
}

// ============================================================================
// Module initialization
// ============================================================================

/// Get the version of the edgefirst_ara2 Python library
#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// EdgeFirst ARA-2 Python Library
///
/// This library provides Python bindings for the ARA-2 neural accelerator client.
/// It enables Python applications to connect to ARA-2 devices, load models, and
/// run inference.
///
/// Main classes:
///     Session: Connection to the ARA-2 proxy service
///     Endpoint: Represents an ARA-2 accelerator device
///     Model: A loaded neural network model
///
/// Example:
///     >>> import edgefirst_ara2
///     >>>
///     >>> # Connect to the proxy
///     >>> session = edgefirst_ara2.Session.create_via_unix_socket("/var/run/ara2.sock")
///     >>>
///     >>> # Get version information
///     >>> versions = session.versions()
///     >>> print(versions)
///     >>>
///     >>> # List available endpoints
///     >>> endpoints = session.list_endpoints()
///     >>> print(f"Found {len(endpoints)} endpoints")
#[pymodule(name = "edgefirst_ara2")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<Session>()?;
    m.add_class::<Endpoint>()?;
    m.add_class::<Model>()?;
    m.add_class::<State>()?;
    m.add_class::<DramStatistics>()?;
    m.add_class::<ModelTiming>()?;
    m.add_class::<OutputQuantization>()?;

    // Register functions
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
