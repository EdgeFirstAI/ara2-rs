//! Finding a running ARA-2 proxy, and reading where one is configured to
//! listen.
//!
//! Neither the socket path nor the transport is fixed: they come from the
//! proxy's YAML configuration, which differs between packagings, and the
//! path to that configuration is itself a command-line argument. So rather
//! than assume any of it, [`discover`] observes the sockets a running proxy
//! actually holds open and falls back to reading the configuration it was
//! started with.

use crate::{Session, error::Error};
use serde::Deserialize;
use std::{
    fs,
    net::Ipv4Addr,
    path::{Path, PathBuf},
};

/// The `comm` of the ARA-2 proxy's main thread, matched by [`discover`].
///
/// Both packagings use it despite shipping differently named executables
/// (`proxy_ara240` under NXP's rt-sdk-ara2, `proxy` under the Kinara
/// runtime meta-kinara packages), which makes it the one handle common to
/// both. `comm` is capped at 15 characters by the kernel.
pub const PROXY_COMM: &str = "kinara_main";

/// An address a proxy accepts connections on.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProxyEndpoint {
    /// A filesystem-visible UNIX socket.
    Unix(PathBuf),
    /// A TCP/IPv4 host and port.
    Tcp(Ipv4Addr, u16),
}

impl std::fmt::Display for ProxyEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyEndpoint::Unix(path) => write!(f, "unix:{}", path.display()),
            ProxyEndpoint::Tcp(ip, port) => write!(f, "tcp:{ip}:{port}"),
        }
    }
}

impl ProxyEndpoint {
    /// Open a session against this endpoint.
    pub fn connect(&self) -> Result<Session, Error> {
        match self {
            ProxyEndpoint::Unix(path) => Session::create_via_unix_socket(&path.to_string_lossy()),
            ProxyEndpoint::Tcp(ip, port) => Session::create_via_tcp_ipv4_socket(*ip, *port),
        }
    }
}

/// A running ARA-2 proxy found by [`discover`].
///
/// `endpoints` is empty when neither route to them worked -- the process
/// is still reported, because knowing a proxy is running but unreachable
/// is more useful than not knowing it is there.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Proxy {
    /// Process ID.
    pub pid: u32,
    /// Where it accepts connections.
    pub endpoints: Vec<ProxyEndpoint>,
    /// The configuration file named on its command line, if it named one.
    pub config: Option<PathBuf>,
    /// Its executable, if readable.
    pub exe: Option<PathBuf>,
}

impl Proxy {
    /// Open a session against this proxy's first endpoint.
    pub fn connect(&self) -> Result<Session, Error> {
        self.endpoints
            .first()
            .ok_or(Error::NoProxyEndpoint {
                pid: Some(self.pid),
            })?
            .connect()
    }
}

/// The `proxy:` section of a proxy configuration file, as far as it
/// concerns clients.
///
/// The three address fields are lists, and `interface_type` selects which
/// one is live; the others are left in the file as documentation.
#[derive(Debug, Deserialize)]
struct RawConfig {
    proxy: RawProxy,
}

#[derive(Debug, Deserialize)]
struct RawProxy {
    #[serde(default)]
    interface_type: Option<String>,
    #[serde(default)]
    interface_socket_file: Vec<PathBuf>,
    #[serde(default)]
    interface_ip_address: Vec<String>,
    #[serde(default)]
    interface_port: Vec<u16>,
}

/// The endpoints a proxy configuration file declares.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ProxyConfig {
    /// The file this was read from.
    pub path: PathBuf,
    /// Endpoints declared by the interface type in force.
    pub endpoints: Vec<ProxyEndpoint>,
}

impl ProxyConfig {
    /// Read and parse a proxy configuration file.
    ///
    /// `interface_type` decides which address list is in force: `SOCKET`
    /// takes `interface_socket_file`, `IPV4` pairs `interface_ip_address`
    /// with `interface_port` positionally. `NAMED_PIPE` is a Windows
    /// transport and yields no endpoints here.
    pub fn read(path: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(path).map_err(|source| Error::ProxyConfigUnreadable {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parse configuration text already in hand.
    fn parse(text: &str, path: &Path) -> Result<Self, Error> {
        let raw: RawConfig =
            serde_yaml_ng::from_str(text).map_err(|e| Error::ProxyConfigInvalid {
                path: path.to_path_buf(),
                reason: e.to_string(),
            })?;

        // Default to SOCKET rather than erroring on an absent
        // interface_type: it is the only transport either packaging ships
        // configured on Linux, and a file that omits it is still usable.
        let kind = raw
            .proxy
            .interface_type
            .as_deref()
            .unwrap_or("SOCKET")
            .to_ascii_uppercase();

        let endpoints = match kind.as_str() {
            "SOCKET" => raw
                .proxy
                .interface_socket_file
                .into_iter()
                .map(ProxyEndpoint::Unix)
                .collect(),
            "IPV4" => raw
                .proxy
                .interface_ip_address
                .iter()
                .zip(raw.proxy.interface_port.iter())
                .filter_map(|(ip, port)| ip.parse().ok().map(|ip| ProxyEndpoint::Tcp(ip, *port)))
                .collect(),
            "NAMED_PIPE" => Vec::new(),
            other => {
                return Err(Error::ProxyConfigInvalid {
                    path: path.to_path_buf(),
                    reason: format!("unknown interface_type {other:?}"),
                });
            }
        };

        Ok(ProxyConfig {
            path: path.to_path_buf(),
            endpoints,
        })
    }
}

/// Every ARA-2 proxy currently running on this host.
///
/// Returns one entry per process whose `comm` is [`PROXY_COMM`]. More than
/// one can be running, so this is a list rather than an option; an empty
/// list means none was found, which is not an error.
///
/// Endpoints are resolved by observing the sockets the process holds open,
/// which reflects where it is really listening however that was decided.
/// That needs permission to read the process's file descriptors -- the
/// proxy runs as root on both packagings -- so when it is unavailable this
/// falls back to parsing the configuration file named on the command line.
pub fn discover() -> Result<Vec<Proxy>, Error> {
    let mut found = Vec::new();

    let entries = fs::read_dir("/proc").map_err(|source| Error::ProcUnavailable {
        path: PathBuf::from("/proc"),
        source,
    })?;

    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };

        // A process can exit at any point during this walk, so every read
        // below treats absence as "not a proxy" rather than an error.
        let comm = match fs::read_to_string(format!("/proc/{pid}/comm")) {
            Ok(comm) => comm,
            Err(_) => continue,
        };
        if comm.trim() != PROXY_COMM {
            continue;
        }

        let cmdline = read_cmdline(pid);
        let config = config_arg(&cmdline);
        let exe = fs::read_link(format!("/proc/{pid}/exe")).ok();

        let mut endpoints = observed_endpoints(pid);
        if endpoints.is_empty()
            && let Some(path) = config.as_deref()
            && let Ok(parsed) = ProxyConfig::read(path)
        {
            endpoints = parsed.endpoints;
        }

        found.push(Proxy {
            pid,
            endpoints,
            config,
            exe,
        });
    }

    found.sort_by_key(|proxy| proxy.pid);
    Ok(found)
}

/// Arguments of `/proc/<pid>/cmdline`, which are NUL-separated.
fn read_cmdline(pid: u32) -> Vec<String> {
    fs::read(format!("/proc/{pid}/cmdline"))
        .map(|raw| {
            raw.split(|b| *b == 0)
                .filter(|arg| !arg.is_empty())
                .map(|arg| String::from_utf8_lossy(arg).into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// The configuration path from `-c PATH`, `--config PATH` or
/// `--config=PATH`.
///
/// The two packagings spell the flag differently, and neither spelling is
/// guaranteed to be present at all, so a missing path is normal.
fn config_arg(cmdline: &[String]) -> Option<PathBuf> {
    let mut args = cmdline.iter();
    while let Some(arg) = args.next() {
        if let Some(path) = arg.strip_prefix("--config=") {
            return Some(PathBuf::from(path));
        }
        if arg == "-c" || arg == "--config" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

/// Endpoints derived from the listening sockets a process holds open.
///
/// Returns empty rather than failing when the file descriptors cannot be
/// read, which is the ordinary case for a non-root caller inspecting the
/// root-owned proxy.
fn observed_endpoints(pid: u32) -> Vec<ProxyEndpoint> {
    let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else {
        return Vec::new();
    };

    let inodes: Vec<u64> = fds
        .flatten()
        .filter_map(|fd| fs::read_link(fd.path()).ok())
        .filter_map(|target| {
            target
                .to_str()?
                .strip_prefix("socket:[")?
                .strip_suffix(']')?
                .parse()
                .ok()
        })
        .collect();
    if inodes.is_empty() {
        return Vec::new();
    }

    let mut endpoints = unix_listeners(&inodes);
    endpoints.extend(tcp_listeners(&inodes));
    endpoints
}

/// Listening UNIX sockets among `inodes`, from `/proc/net/unix`.
///
/// Columns are `Num RefCount Protocol Flags Type St Inode Path`. A
/// listening socket is identified by `SO_ACCEPTCON` in `Flags`, not by
/// `St`, which is also `01` for sockets that are merely unconnected.
fn unix_listeners(inodes: &[u64]) -> Vec<ProxyEndpoint> {
    const SO_ACCEPTCON: u64 = 0x1_0000;

    let Ok(text) = fs::read_to_string("/proc/net/unix") else {
        return Vec::new();
    };

    text.lines()
        .filter_map(|line| {
            let mut cols = line.split_whitespace();
            let flags = u64::from_str_radix(cols.nth(3)?, 16).ok()?;
            let inode: u64 = cols.nth(2)?.parse().ok()?;
            let path = cols.next()?;
            (flags & SO_ACCEPTCON != 0 && inodes.contains(&inode) && path.starts_with('/'))
                .then(|| ProxyEndpoint::Unix(PathBuf::from(path)))
        })
        .collect()
}

/// Listening TCP sockets among `inodes`, from `/proc/net/tcp`.
///
/// Columns are `sl local_address rem_address st ... inode`, with the
/// address as little-endian hex and state `0A` for `TCP_LISTEN`.
fn tcp_listeners(inodes: &[u64]) -> Vec<ProxyEndpoint> {
    const TCP_LISTEN: &str = "0A";

    let Ok(text) = fs::read_to_string("/proc/net/tcp") else {
        return Vec::new();
    };

    text.lines()
        .skip(1)
        .filter_map(|line| {
            let mut cols = line.split_whitespace();
            let local = cols.nth(1)?;
            if cols.next()? != TCP_LISTEN {
                return None;
            }
            let inode: u64 = cols.nth(5)?.parse().ok()?;
            if !inodes.contains(&inode) {
                return None;
            }
            let (addr, port) = local.split_once(':')?;
            let addr = u32::from_str_radix(addr, 16).ok()?;
            let port = u16::from_str_radix(port, 16).ok()?;
            Some(ProxyEndpoint::Tcp(Ipv4Addr::from(addr.to_be()), port))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_arg_accepts_both_spellings() {
        let short = [
            "./proxy_ara240",
            "-c",
            "/etc/rt-sdk-ara240/proxy_config.yaml",
        ]
        .map(String::from);
        let long = ["/usr/libexec/ara2/proxy", "--config", "/etc/ara2.yaml"].map(String::from);
        let joined = ["proxy", "--config=/etc/ara2.yaml"].map(String::from);

        assert_eq!(
            config_arg(&short),
            Some(PathBuf::from("/etc/rt-sdk-ara240/proxy_config.yaml"))
        );
        assert_eq!(config_arg(&long), Some(PathBuf::from("/etc/ara2.yaml")));
        assert_eq!(config_arg(&joined), Some(PathBuf::from("/etc/ara2.yaml")));
        assert_eq!(config_arg(&["proxy".to_owned()]), None);
        assert_eq!(config_arg(&["proxy".to_owned(), "-c".to_owned()]), None);
    }

    fn parse(yaml: &str) -> Result<Vec<ProxyEndpoint>, Error> {
        ProxyConfig::parse(yaml, Path::new("proxy_config.yaml")).map(|cfg| cfg.endpoints)
    }

    #[test]
    fn socket_config_yields_unix_endpoints() {
        let endpoints = parse(
            r#"
proxy:
  interface_type: "SOCKET"
  interface_ip_address: ["127.0.0.1"]
  interface_port: [5000]
  interface_socket_file: ["/var/run/proxy.sock"]
"#,
        )
        .unwrap();
        assert_eq!(
            endpoints,
            vec![ProxyEndpoint::Unix(PathBuf::from("/var/run/proxy.sock"))]
        );
    }

    /// The address lists stay in the file when another transport is in
    /// force, so the type has to select rather than the presence of a list.
    #[test]
    fn ipv4_config_pairs_addresses_with_ports() {
        let endpoints = parse(
            r#"
proxy:
  interface_type: "IPV4"
  interface_ip_address: ["127.0.0.1", "10.0.0.2"]
  interface_port: [5000, 5001]
  interface_socket_file: ["/var/run/proxy.sock"]
"#,
        )
        .unwrap();
        assert_eq!(
            endpoints,
            vec![
                ProxyEndpoint::Tcp(Ipv4Addr::new(127, 0, 0, 1), 5000),
                ProxyEndpoint::Tcp(Ipv4Addr::new(10, 0, 0, 2), 5001),
            ]
        );
    }

    #[test]
    fn absent_interface_type_defaults_to_socket() {
        let endpoints = parse(
            r#"
proxy:
  interface_socket_file: ["/var/run/ara2.sock"]
"#,
        )
        .unwrap();
        assert_eq!(
            endpoints,
            vec![ProxyEndpoint::Unix(PathBuf::from("/var/run/ara2.sock"))]
        );
    }

    #[test]
    fn unknown_interface_type_is_an_error() {
        let err = parse(
            r#"
proxy:
  interface_type: "CARRIER_PIGEON"
  interface_socket_file: ["/var/run/ara2.sock"]
"#,
        )
        .unwrap_err();
        assert!(matches!(err, Error::ProxyConfigInvalid { .. }));
    }

    #[test]
    fn named_pipe_config_yields_no_endpoints() {
        assert!(
            parse(
                r#"
proxy:
  interface_type: "NAMED_PIPE"
  interface_socket_file: ["/var/run/ara2.sock"]
"#,
            )
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn endpoint_display_names_the_transport() {
        assert_eq!(
            ProxyEndpoint::Unix(PathBuf::from("/var/run/proxy.sock")).to_string(),
            "unix:/var/run/proxy.sock"
        );
        assert_eq!(
            ProxyEndpoint::Tcp(Ipv4Addr::new(127, 0, 0, 1), 5000).to_string(),
            "tcp:127.0.0.1:5000"
        );
    }
}
