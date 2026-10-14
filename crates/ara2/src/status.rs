//! Status codes and the detailed status DVAPI 1.3+ attaches to them.

/// Which part of the stack a detailed status attributes a failure to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatusComponent {
    /// The client library in this process.
    ClientLib,
    /// The proxy service.
    Proxy,
    /// The NPU device.
    Device,
    /// The NPU's memory.
    DeviceMemory,
    /// A value this build does not name.
    Other(u32),
}

impl StatusComponent {
    pub(crate) fn from_raw(value: u32) -> Self {
        match value {
            0 => StatusComponent::ClientLib,
            1 => StatusComponent::Proxy,
            2 => StatusComponent::Device,
            3 => StatusComponent::DeviceMemory,
            other => StatusComponent::Other(other),
        }
    }
}

/// How serious a detailed status says a failure is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Severity {
    /// Fully functional.
    Ok,
    /// Temporary; retrying may succeed.
    Retryable,
    /// Functional, but not as expected.
    Degraded,
    /// Not recoverable.
    Fatal,
    /// A value this build does not name.
    Other(u32),
}

impl Severity {
    pub(crate) fn from_raw(value: u32) -> Self {
        match value {
            0 => Severity::Ok,
            1 => Severity::Retryable,
            2 => Severity::Degraded,
            3 => Severity::Fatal,
            other => Severity::Other(other),
        }
    }
}

/// The detail DVAPI 1.3+ returns alongside a status code from its
/// detailed-status calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatusDetail {
    /// Component-specific sub-code.
    pub sub_code: u32,
    /// Where the failure arose.
    pub component: StatusComponent,
    /// How serious it is.
    pub severity: Severity,
}

/// A `dv_status_t` read out of either version's struct.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawStatus {
    pub code: u32,
    pub sub_code: u32,
    pub component: u32,
    pub severity: u32,
}

impl std::fmt::Display for StatusComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatusComponent::ClientLib => f.write_str("client library"),
            StatusComponent::Proxy => f.write_str("proxy"),
            StatusComponent::Device => f.write_str("device"),
            StatusComponent::DeviceMemory => f.write_str("device memory"),
            StatusComponent::Other(v) => write!(f, "component {v}"),
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Ok => f.write_str("ok"),
            Severity::Retryable => f.write_str("retryable"),
            Severity::Degraded => f.write_str("degraded"),
            Severity::Fatal => f.write_str("fatal"),
            Severity::Other(v) => write!(f, "severity {v}"),
        }
    }
}

impl From<ara2_sys::v1_3::dv_status> for RawStatus {
    fn from(s: ara2_sys::v1_3::dv_status) -> Self {
        RawStatus {
            code: s.main_code,
            sub_code: s.sub_code,
            component: s.type_,
            severity: s.criticality,
        }
    }
}

impl From<ara2_sys::v3_0::dv_status> for RawStatus {
    fn from(s: ara2_sys::v3_0::dv_status) -> Self {
        RawStatus {
            code: s.main_code,
            sub_code: s.sub_code,
            component: s.type_,
            severity: s.criticality,
        }
    }
}

/// The library's own name for `code`.
pub(crate) fn status_name(lib: &ara2_sys::araclient, code: ara2_sys::dv_status_code_t) -> String {
    let Ok(stringify) = lib.dv_stringify_status_code.as_ref() else {
        return format!("status {code}");
    };
    // SAFETY: the library returns a static string, or the literal
    // "DV_STATUS_CODE_UNKNOWN_L" for codes it does not know.
    let ptr = unsafe { stringify(code) };
    if ptr.is_null() {
        return format!("status {code}");
    }
    // SAFETY: non-null, and points at a NUL-terminated static string.
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// A failing status code as an [`Error`](crate::Error), named by `lib`.
/// For use before a session exists; afterwards use
/// `SessionInner::status_error`.
pub(crate) fn status_error_from(
    lib: &ara2_sys::araclient,
    code: ara2_sys::dv_status_code_t,
) -> crate::Error {
    crate::Error::Ara2 {
        code: code as i32,
        name: status_name(lib, code),
        detail: None,
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;

    #[test]
    fn severity_and_component_map_known_values_and_keep_unknown() {
        assert_eq!(Severity::from_raw(1), Severity::Retryable);
        assert_eq!(Severity::from_raw(3), Severity::Fatal);
        assert_eq!(Severity::from_raw(9), Severity::Other(9));
        assert_eq!(StatusComponent::from_raw(1), StatusComponent::Proxy);
        assert_eq!(StatusComponent::from_raw(7), StatusComponent::Other(7));
    }

    #[test]
    fn display_names_the_code_and_detail() {
        let e = crate::Error::Ara2 {
            code: 223,
            name: "DV_SESSION_INVALID_HANDLE".to_owned(),
            detail: Some(StatusDetail {
                sub_code: 4,
                component: StatusComponent::Proxy,
                severity: Severity::Retryable,
            }),
        };
        let text = e.to_string();
        assert!(text.contains("DV_SESSION_INVALID_HANDLE (223)"), "{text}");
        assert!(text.contains("proxy"), "{text}");
        assert!(text.contains("retryable"), "{text}");
    }
}
