//! Component versions reported by the proxy.

use ara2_sys::{
    DV_PRODUCT_TYPE_CLIENT_LIB, DV_PRODUCT_TYPE_CNN_MODEL, DV_PRODUCT_TYPE_FIRMWARE,
    DV_PRODUCT_TYPE_LLM_MODEL, DV_PRODUCT_TYPE_PCI_DRIVER, DV_PRODUCT_TYPE_PROXY,
    DV_PRODUCT_TYPE_SYSAPI, dv_product_version, dv_version,
    v3_0::dv_product_type_3_0_MCS_LLM_MODEL,
};

use crate::Version;

/// A software or firmware component the proxy reports a version for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Component {
    /// The proxy service.
    Proxy,
    /// The PCIe driver.
    PciDriver,
    /// Endpoint firmware.
    Firmware,
    /// The CNN model format.
    CnnModel,
    /// The LLM model format.
    LlmModel,
    /// The multi-chip-sharded LLM model format (DVAPI 3.0).
    McsLlmModel,
    /// The client library.
    ClientLib,
    /// The system API.
    SysApi,
    /// A product type this build does not name; carries the raw value.
    Other(i32),
}

impl Component {
    pub(crate) fn from_raw(value: i32) -> Self {
        const MCS: i32 = dv_product_type_3_0_MCS_LLM_MODEL as i32;
        match value {
            v if v == DV_PRODUCT_TYPE_PROXY => Component::Proxy,
            v if v == DV_PRODUCT_TYPE_PCI_DRIVER => Component::PciDriver,
            v if v == DV_PRODUCT_TYPE_FIRMWARE => Component::Firmware,
            v if v == DV_PRODUCT_TYPE_CNN_MODEL => Component::CnnModel,
            v if v == DV_PRODUCT_TYPE_LLM_MODEL => Component::LlmModel,
            v if v == DV_PRODUCT_TYPE_CLIENT_LIB => Component::ClientLib,
            v if v == DV_PRODUCT_TYPE_SYSAPI => Component::SysApi,
            MCS => Component::McsLlmModel,
            other => Component::Other(other),
        }
    }
}

impl std::fmt::Display for Component {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Component::Proxy => f.write_str("proxy"),
            Component::PciDriver => f.write_str("pci_driver"),
            Component::Firmware => f.write_str("firmware"),
            Component::CnnModel => f.write_str("cnn_model"),
            Component::LlmModel => f.write_str("llm_model"),
            Component::McsLlmModel => f.write_str("mcs_llm_model"),
            Component::ClientLib => f.write_str("client_lib"),
            Component::SysApi => f.write_str("sysapi"),
            Component::Other(v) => write!(f, "product_type_{v}"),
        }
    }
}

/// A component and its version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentVersion {
    /// Which component.
    pub component: Component,
    /// Its version.
    pub version: Version,
}

pub(crate) fn component_versions(list: &[dv_product_version]) -> Vec<ComponentVersion> {
    list.iter()
        .map(|p| ComponentVersion {
            component: Component::from_raw(p.product_type),
            version: from_raw(p.product_version),
        })
        .collect()
}

pub(crate) fn from_raw(v: dv_version) -> Version {
    Version {
        major: v.major,
        minor: v.minor,
        patch: v.patch,
        patch_minor: v.patch_minor,
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;

    fn raw(product_type: i32, major: u8) -> dv_product_version {
        dv_product_version {
            product_type: product_type as _,
            product_version: dv_version {
                major,
                minor: 0,
                patch: 0,
                patch_minor: 0,
            },
        }
    }

    #[test]
    fn known_product_types_map_to_components() {
        assert_eq!(Component::from_raw(0), Component::Proxy);
        assert_eq!(Component::from_raw(2), Component::Firmware);
        assert_eq!(Component::from_raw(5), Component::ClientLib);
        assert_eq!(Component::from_raw(7), Component::McsLlmModel);
    }

    #[test]
    fn unknown_product_types_are_kept_not_errors() {
        assert_eq!(Component::from_raw(42), Component::Other(42));
        assert_eq!(Component::from_raw(-1), Component::Other(-1));
    }

    #[test]
    fn duplicates_are_kept_in_order() {
        let list = [raw(2, 2), raw(2, 3), raw(0, 3)];
        let out = component_versions(&list);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].component, Component::Firmware);
        assert_eq!(out[1].version.major, 3);
        assert_eq!(out[2].component, Component::Proxy);
    }
}
