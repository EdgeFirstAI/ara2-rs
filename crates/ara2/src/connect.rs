//! Choosing which proxy endpoint `Session::connect` opens.
//!
//! Exactly one connection attempt is made, against an endpoint chosen by
//! inspection: a failed attempt is not free, because the DVAPI 1.1 client
//! keeps state across one and a later create can come back
//! `DV_SESSION_INVALID_HANDLE`.

use std::net::Ipv4Addr;

use crate::{Error, Proxy, ProxyEndpoint, SOCKET_OVERRIDE_ENV, SOCKET_PATHS};

/// Parses an `ARA2_SOCKET` value: `unix:/path`, `tcp:IPV4:PORT`, or a bare
/// UNIX socket path.
pub(crate) fn parse_endpoint(s: &str) -> Result<ProxyEndpoint, Error> {
    let invalid = || Error::InvalidSocketOverride(s.to_owned());
    if let Some(addr) = s.strip_prefix("tcp:") {
        let (ip, port) = addr.rsplit_once(':').ok_or_else(invalid)?;
        let ip: Ipv4Addr = ip.parse().map_err(|_| invalid())?;
        let port: u16 = port.parse().map_err(|_| invalid())?;
        return Ok(ProxyEndpoint::Tcp(ip, port));
    }
    let path = s.strip_prefix("unix:").unwrap_or(s);
    if path.is_empty() {
        return Err(invalid());
    }
    Ok(ProxyEndpoint::Unix(path.into()))
}

/// Picks the endpoint to connect to: the override if given, else the first
/// existing path in `existing`, else a discovered proxy's UNIX endpoint,
/// else a discovered proxy's TCP endpoint.
pub(crate) fn choose_endpoint(
    override_: Option<&str>,
    existing: &[&str],
    proxies: &[Proxy],
) -> Result<ProxyEndpoint, Error> {
    if let Some(value) = override_ {
        return parse_endpoint(value);
    }
    if let Some(path) = existing.first() {
        return Ok(ProxyEndpoint::Unix((*path).into()));
    }
    let all = || proxies.iter().flat_map(|p| p.endpoints.iter());
    if let Some(unix) = all().find(|e| matches!(e, ProxyEndpoint::Unix(_))) {
        return Ok(unix.clone());
    }
    if let Some(tcp) = all().next() {
        return Ok(tcp.clone());
    }
    let mut checked: Vec<String> = SOCKET_PATHS.iter().map(|p| format!("unix:{p}")).collect();
    checked.extend(
        proxies
            .iter()
            .map(|p| format!("proxy pid {} (no reachable endpoint)", p.pid)),
    );
    if proxies.is_empty() {
        checked.push("no running proxy found".to_owned());
    }
    Err(Error::NoProxyFound { checked })
}

/// Resolves the endpoint [`crate::Session::connect`] will open, without
/// opening it.
///
/// Resolution order: `ARA2_SOCKET`, then the first existing entry of
/// [`SOCKET_PATHS`], then a running proxy found by [`crate::discover`]
/// (UNIX endpoints before TCP).
pub fn proxy_endpoint() -> Result<ProxyEndpoint, Error> {
    let override_ = std::env::var(SOCKET_OVERRIDE_ENV).ok();
    let existing: Vec<&str> = SOCKET_PATHS
        .iter()
        .copied()
        .filter(|p| std::path::Path::new(p).exists())
        .collect();
    let proxies = if override_.is_none() && existing.is_empty() {
        crate::discover().unwrap_or_default()
    } else {
        Vec::new()
    };
    choose_endpoint(override_.as_deref(), &existing, &proxies)
}

#[cfg(test)]
mod offline_tests {
    use super::*;
    use crate::Proxy;
    use std::net::Ipv4Addr;
    use std::path::PathBuf;

    fn proxy(endpoints: Vec<ProxyEndpoint>) -> Proxy {
        Proxy {
            pid: 1,
            endpoints,
            config: None,
            exe: None,
        }
    }

    #[test]
    fn parses_unix_tcp_and_bare_paths() {
        assert_eq!(
            parse_endpoint("/run/ara2.sock").unwrap(),
            ProxyEndpoint::Unix("/run/ara2.sock".into())
        );
        assert_eq!(
            parse_endpoint("unix:/run/proxy.sock").unwrap(),
            ProxyEndpoint::Unix("/run/proxy.sock".into())
        );
        assert_eq!(
            parse_endpoint("tcp:127.0.0.1:5000").unwrap(),
            ProxyEndpoint::Tcp(Ipv4Addr::LOCALHOST, 5000)
        );
    }

    #[test]
    fn rejects_malformed_tcp_instead_of_treating_it_as_a_path() {
        for bad in [
            "tcp:127.0.0.1",
            "tcp:localhost:5000",
            "tcp:127.0.0.1:99999",
            "tcp:",
            "unix:",
        ] {
            assert!(
                matches!(parse_endpoint(bad), Err(Error::InvalidSocketOverride(_))),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn override_wins() {
        let chosen =
            choose_endpoint(Some("tcp:10.0.0.2:6000"), &["/var/run/proxy.sock"], &[]).unwrap();
        assert_eq!(chosen, ProxyEndpoint::Tcp(Ipv4Addr::new(10, 0, 0, 2), 6000));
    }

    #[test]
    fn existing_socket_path_beats_discovery() {
        let tcp = proxy(vec![ProxyEndpoint::Tcp(Ipv4Addr::LOCALHOST, 5000)]);
        let chosen = choose_endpoint(None, &["/var/run/ara2.sock"], &[tcp]).unwrap();
        assert_eq!(chosen, ProxyEndpoint::Unix("/var/run/ara2.sock".into()));
    }

    #[test]
    fn discovery_prefers_unix_over_tcp_across_proxies() {
        let tcp = proxy(vec![ProxyEndpoint::Tcp(Ipv4Addr::LOCALHOST, 5000)]);
        let unix = proxy(vec![ProxyEndpoint::Unix(PathBuf::from("/srv/p.sock"))]);
        let chosen = choose_endpoint(None, &[], &[tcp, unix]).unwrap();
        assert_eq!(chosen, ProxyEndpoint::Unix("/srv/p.sock".into()));
    }

    #[test]
    fn discovery_falls_back_to_tcp() {
        let tcp = proxy(vec![ProxyEndpoint::Tcp(Ipv4Addr::LOCALHOST, 5000)]);
        assert_eq!(
            choose_endpoint(None, &[], &[tcp]).unwrap(),
            ProxyEndpoint::Tcp(Ipv4Addr::LOCALHOST, 5000)
        );
    }

    #[test]
    fn nothing_found_lists_what_was_checked() {
        match choose_endpoint(None, &[], &[proxy(vec![])]) {
            Err(Error::NoProxyFound { checked }) => {
                assert!(checked.iter().any(|c| c.contains("/var/run/proxy.sock")));
                assert!(checked.iter().any(|c| c.contains("pid 1")));
            }
            other => panic!("expected NoProxyFound, got {other:?}"),
        }
    }
}
