// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::error::to_py_err;
use crate::session::Session;
use pyo3::prelude::*;
use std::path::PathBuf;

/// An address a proxy accepts connections on.
///
/// ``kind`` is ``"unix"`` or ``"tcp"``. ``path`` is set for the former,
/// ``host`` and ``port`` for the latter.
#[pyclass(module = "edgefirst_ara2", from_py_object)]
#[derive(Clone)]
pub struct ProxyEndpoint(pub(crate) ara2::ProxyEndpoint);

#[pymethods]
impl ProxyEndpoint {
    /// Transport: ``"unix"`` or ``"tcp"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0 {
            ara2::ProxyEndpoint::Unix(_) => "unix",
            ara2::ProxyEndpoint::Tcp(..) => "tcp",
            _ => "unknown",
        }
    }

    /// Socket path, for a ``"unix"`` endpoint; ``None`` otherwise.
    #[getter]
    fn path(&self) -> Option<PathBuf> {
        match &self.0 {
            ara2::ProxyEndpoint::Unix(path) => Some(path.clone()),
            _ => None,
        }
    }

    /// IPv4 address, for a ``"tcp"`` endpoint; ``None`` otherwise.
    #[getter]
    fn host(&self) -> Option<String> {
        match &self.0 {
            ara2::ProxyEndpoint::Tcp(ip, _) => Some(ip.to_string()),
            _ => None,
        }
    }

    /// TCP port, for a ``"tcp"`` endpoint; ``None`` otherwise.
    #[getter]
    fn port(&self) -> Option<u16> {
        match &self.0 {
            ara2::ProxyEndpoint::Tcp(_, port) => Some(*port),
            _ => None,
        }
    }

    /// Open a session against this endpoint.
    fn connect(&self) -> PyResult<Session> {
        Ok(Session(Some(self.0.connect().map_err(to_py_err)?)))
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ProxyEndpoint({})", self.0)
    }
}

/// A running ARA-2 proxy found by :func:`discover`.
#[pyclass(module = "edgefirst_ara2", from_py_object)]
#[derive(Clone)]
pub struct Proxy(pub(crate) ara2::Proxy);

#[pymethods]
impl Proxy {
    /// Process ID of the running proxy.
    #[getter]
    fn pid(&self) -> u32 {
        self.0.pid
    }

    /// Where it accepts connections. Empty when neither reading its file
    /// descriptors nor its configuration file was possible.
    #[getter]
    fn endpoints(&self) -> Vec<ProxyEndpoint> {
        self.0
            .endpoints
            .iter()
            .cloned()
            .map(ProxyEndpoint)
            .collect()
    }

    /// The configuration file named on its command line, if it named one.
    #[getter]
    fn config(&self) -> Option<PathBuf> {
        self.0.config.clone()
    }

    /// Its executable, if readable.
    #[getter]
    fn exe(&self) -> Option<PathBuf> {
        self.0.exe.clone()
    }

    /// Open a session against this proxy's first endpoint.
    fn connect(&self) -> PyResult<Session> {
        Ok(Session(Some(self.0.connect().map_err(to_py_err)?)))
    }

    fn __repr__(&self) -> String {
        let endpoints: Vec<String> = self.0.endpoints.iter().map(|e| e.to_string()).collect();
        format!(
            "Proxy(pid={}, endpoints=[{}])",
            self.0.pid,
            endpoints.join(", ")
        )
    }
}

/// Find every ARA-2 proxy currently running on this host.
///
/// Returns a list because more than one proxy can be running; an empty
/// list means none was found, which is not an error.
///
/// Endpoints are resolved by observing the sockets each process holds
/// open, which reflects where it is really listening however that was
/// decided. That needs permission to read the process's file descriptors
/// -- the proxy runs as root under both packagings -- so when it is
/// unavailable this falls back to parsing the configuration file named on
/// the command line.
///
/// Example:
///     >>> import edgefirst_ara2
///     >>> for proxy in edgefirst_ara2.discover():
///     ...     print(proxy.pid, [str(e) for e in proxy.endpoints])
#[pyfunction]
pub fn discover() -> PyResult<Vec<Proxy>> {
    Ok(ara2::discover()
        .map_err(to_py_err)?
        .into_iter()
        .map(Proxy)
        .collect())
}
