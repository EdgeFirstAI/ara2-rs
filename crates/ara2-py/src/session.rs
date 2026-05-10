// SPDX-License-Identifier: Apache-2.0
// Copyright © 2025 Au-Zone Technologies. All Rights Reserved.

use crate::endpoint::Endpoint;
use crate::error::to_py_err;
use pyo3::prelude::*;
use std::{collections::HashMap, net::Ipv4Addr, path::PathBuf, str::FromStr};

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
///     >>> session.close()
#[pyclass(module = "edgefirst_ara2")]
pub struct Session(pub(crate) Option<ara2::Session>);

impl Session {
    fn inner(&self) -> PyResult<&ara2::Session> {
        self.0
            .as_ref()
            .ok_or_else(|| crate::error::Ara2Error::new_err("session is closed"))
    }
}

#[pymethods]
impl Session {
    /// Create a session connected via UNIX domain socket.
    ///
    /// Args:
    ///     socket_path: Path to the UNIX socket (str or os.PathLike,
    ///                  e.g., "/var/run/ara2.sock")
    ///
    /// Returns:
    ///     Session: A new session connected to the proxy
    ///
    /// Raises:
    ///     ProxyError: If the socket does not exist or the proxy is not running
    #[staticmethod]
    fn create_via_unix_socket(socket_path: PathBuf) -> PyResult<Self> {
        let path_str = socket_path.to_string_lossy();
        Ok(Session(Some(
            ara2::Session::create_via_unix_socket(&path_str).map_err(to_py_err)?,
        )))
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
        Ok(Session(Some(
            ara2::Session::create_via_tcp_ipv4_socket(ip_addr, port).map_err(to_py_err)?,
        )))
    }

    /// Get version information for all components.
    ///
    /// Returns:
    ///     dict: Dictionary mapping component names to version strings
    fn versions(&self) -> PyResult<HashMap<String, String>> {
        self.inner()?.versions().map_err(to_py_err)
    }

    /// List all available endpoints.
    ///
    /// Returns:
    ///     list[Endpoint]: List of available ARA-2 endpoints
    fn list_endpoints(&self) -> PyResult<Vec<Endpoint>> {
        let endpoints = self.inner()?.list_endpoints().map_err(to_py_err)?;
        Ok(endpoints.into_iter().map(Endpoint).collect())
    }

    /// Get the number of in-flight inference requests for this session.
    ///
    /// Returns the count of requests submitted via :meth:`Model.submit`
    /// that the client library has not yet received a response for.
    ///
    /// Returns:
    ///     int: Number of pending inference requests
    fn inflight_count(&self) -> PyResult<i32> {
        self.inner()?.inflight_count().map_err(to_py_err)
    }

    /// Get the socket type used for this session.
    #[getter]
    fn socket_type(&self) -> PyResult<&str> {
        Ok(match self.inner()?.socket_type() {
            ara2::SocketType::Unix => "unix",
            ara2::SocketType::Tcp => "tcp",
        })
    }

    /// Close this Python session handle.
    ///
    /// After calling ``close()``, any further method call on this
    /// Session raises ``Ara2Error``. This only drops this Python
    /// handle; the underlying proxy connection may remain open while
    /// other objects created from this session (such as ``Endpoint``
    /// or ``Model``) are still alive. The underlying connection is
    /// released when the last handle referencing it is dropped. Safe
    /// to call multiple times.
    fn close(&mut self) {
        self.0 = None;
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            Some(inner) => format!("Session(socket_type={:?})", inner.socket_type()),
            None => "Session(closed)".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[allow(unused_variables)]
    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, pyo3::PyAny>>,
        exc_val: Option<&Bound<'_, pyo3::PyAny>>,
        exc_tb: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> bool {
        self.close();
        false
    }
}
