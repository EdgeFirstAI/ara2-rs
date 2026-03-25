// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::endpoint::Endpoint;
use crate::error::to_py_err;
use pyo3::prelude::*;
use std::{collections::HashMap, net::Ipv4Addr, str::FromStr};

/// ARA-2 session for communicating with the proxy.
///
/// A Session represents a connection to the ARA-2 proxy service, which can
/// be established via either a UNIX socket or TCP socket. The session is used
/// to enumerate endpoints and retrieve version information.
///
/// Example:
///     >>> import edgefirst_ara2
///     >>> session = edgefirst_ara2.Session.create_via_unix_socket("/var/run/ara2.sock")
///     >>> versions = session.versions()
///     >>> endpoints = session.list_endpoints()
#[pyclass(module = "edgefirst_ara2")]
pub struct Session(pub(crate) ara2::Session);

#[pymethods]
impl Session {
    /// Create a session connected via UNIX domain socket.
    ///
    /// Args:
    ///     socket_path: Path to the UNIX socket (e.g., "/var/run/ara2.sock")
    ///
    /// Returns:
    ///     Session: A new session connected to the proxy
    #[staticmethod]
    fn create_via_unix_socket(socket_path: &str) -> PyResult<Self> {
        Ok(Session(
            ara2::Session::create_via_unix_socket(socket_path).map_err(to_py_err)?,
        ))
    }

    /// Create a session connected via TCP/IPv4 socket.
    ///
    /// Args:
    ///     ip: IPv4 address as a string (e.g., "127.0.0.1")
    ///     port: Port number
    ///
    /// Returns:
    ///     Session: A new session connected to the proxy
    #[staticmethod]
    fn create_via_tcp_ipv4_socket(ip: &str, port: u16) -> PyResult<Self> {
        let ip_addr = Ipv4Addr::from_str(ip)
            .map_err(|e| crate::error::Ara2Error::new_err(format!("Invalid IP address: {e}")))?;
        Ok(Session(
            ara2::Session::create_via_tcp_ipv4_socket(ip_addr, port).map_err(to_py_err)?,
        ))
    }

    /// Get version information for all components.
    ///
    /// Returns:
    ///     dict: Dictionary mapping component names to version strings
    fn versions(&self) -> PyResult<HashMap<String, String>> {
        self.0.versions().map_err(to_py_err)
    }

    /// List all available endpoints.
    ///
    /// Returns:
    ///     list[Endpoint]: List of available ARA-2 endpoints
    fn list_endpoints(&self) -> PyResult<Vec<Endpoint>> {
        let endpoints = self.0.list_endpoints().map_err(to_py_err)?;
        Ok(endpoints.into_iter().map(Endpoint).collect())
    }

    /// Get the socket type used for this session.
    #[getter]
    fn socket_type(&self) -> &str {
        match self.0.socket_type() {
            ara2::SocketType::Unix => "unix",
            ara2::SocketType::Tcp => "tcp",
        }
    }

    fn __repr__(&self) -> String {
        format!("Session(socket_type={:?})", self.0.socket_type())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, pyo3::PyAny>>,
        exc_val: Option<&Bound<'_, pyo3::PyAny>>,
        exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> bool {
        false
    }
}
