use std::{
    ffi::{CString, c_char},
    path::Path,
    sync::Arc,
};

use ara2_sys::{
    DV_ENDPOINT_STATE, DV_MODEL_PRIORITY_LEVEL_DV_MODEL_PRIORITY_LEVEL_DEFAULT, dv_endpoint,
    dv_endpoint_dram_statistics, dv_model,
};

use crate::{
    Model,
    error::Error,
    session::{EndpointList, SessionInner},
};

/// The operational state of an NPU endpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl TryFrom<DV_ENDPOINT_STATE> for State {
    type Error = Error;

    fn try_from(value: DV_ENDPOINT_STATE) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => State::Init,
            1 => State::Idle,
            2 => State::Active,
            3 => State::ActiveSlow,
            4 => State::ActiveBoosted,
            5 => State::ThermalInactive,
            6 => State::ThermalUnknown,
            7 => State::Inactive,
            8 => State::Fault,
            _ => return Err(Error::EndpointStateInvalid(value)),
        })
    }
}

/// DRAM usage statistics for an endpoint.
///
/// All sizes are in bytes.
#[derive(Debug, Clone, Copy)]
pub struct DramStatistics {
    /// Total DRAM capacity in bytes.
    pub dram_size: u64,
    /// Total occupied DRAM in bytes.
    pub dram_occupancy_size: u64,
    /// Free DRAM available in bytes.
    pub free_size: u64,
    /// DRAM reserved by the system in bytes.
    pub reserved_occupancy_size: u64,
    /// DRAM occupied by loaded models in bytes.
    pub model_occupancy_size: u64,
    /// DRAM occupied by tensor buffers in bytes.
    pub tensor_occupancy_size: u64,
}

/// An ARA-2 NPU accelerator endpoint.
///
/// Endpoints are cheaply cloneable and keep the session alive through
/// reference counting. The underlying C endpoint list buffer is
/// automatically freed when all endpoints from the same list are dropped.
#[derive(Clone)]
pub struct Endpoint {
    pub(crate) session: Arc<SessionInner>,
    /// Shared ownership of the C-allocated endpoint list buffer.
    pub(crate) _list: Arc<EndpointList>,
    pub(crate) ptr: *mut dv_endpoint,
}

impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint").field("ptr", &self.ptr).finish()
    }
}

// Safety: Endpoint only contains an Arc (Send+Sync) and a raw pointer.
// The pointer is only used for FFI calls which go through the session's
// library handle, which is internally synchronized.
unsafe impl Send for Endpoint {}
unsafe impl Sync for Endpoint {}

impl Endpoint {
    /// Check the current operational state of this endpoint.
    pub fn check_status(&self) -> Result<State, Error> {
        let mut state: DV_ENDPOINT_STATE = 0;
        let err = unsafe {
            self.session
                .lib
                .dv_endpoint_check_status(self.session.ptr, self.ptr, &mut state)
        };

        if err != 0 {
            return Err(err.into());
        }

        State::try_from(state)
    }

    /// Get DRAM usage statistics for this endpoint.
    pub fn dram_statistics(&self) -> Result<DramStatistics, Error> {
        let mut ep_count = 1;
        let mut dram_stats: *mut dv_endpoint_dram_statistics = std::ptr::null_mut();
        let err = unsafe {
            self.session.lib.dv_endpoint_get_dram_statistics(
                self.session.ptr,
                self.ptr,
                &mut dram_stats,
                &mut ep_count,
            )
        };

        if err != 0 {
            return Err(err.into());
        }

        let stats = DramStatistics {
            dram_size: unsafe { (*dram_stats).ep_total_dram_size },
            dram_occupancy_size: unsafe { (*dram_stats).ep_total_dram_occupancy_size },
            free_size: unsafe { (*dram_stats).ep_total_free_size },
            reserved_occupancy_size: unsafe { (*dram_stats).ep_total_reserved_occupancy_size },
            model_occupancy_size: unsafe { (*dram_stats).ep_total_model_occupancy_size },
            tensor_occupancy_size: unsafe { (*dram_stats).ep_total_tensor_occupancy_size },
        };

        unsafe {
            self.session
                .lib
                .dv_endpoint_free_dram_statistics(dram_stats, ep_count);
        }

        Ok(stats)
    }

    /// Load a model from a file onto this endpoint.
    ///
    /// # Arguments
    /// * `path` - Path to the compiled model file (`.dvm`)
    ///
    /// # Returns
    /// A `Model` that can be used for inference. The model keeps the endpoint
    /// (and session) alive through reference counting.
    pub fn load_model_from_file(&self, path: &Path) -> Result<Model, Error> {
        let model_name = match path.file_stem() {
            Some(name) => name.to_string_lossy().to_string(),
            None => "model".to_string(),
        };
        let path_str = path.to_string_lossy().to_string();
        let path_cstr = CString::new(path_str)
            .map_err(|_| Error::NullPointer("Invalid model path".to_owned()))?;
        let name_cstr = CString::new(model_name)
            .map_err(|_| Error::NullPointer("Invalid model name".to_owned()))?;

        let mut model: *mut dv_model = std::ptr::null_mut();
        let err = unsafe {
            self.session.lib.dv_model_load_from_file(
                self.session.ptr,
                self.ptr,
                path_cstr.as_ptr() as *const c_char,
                name_cstr.as_ptr() as *const c_char,
                DV_MODEL_PRIORITY_LEVEL_DV_MODEL_PRIORITY_LEVEL_DEFAULT,
                &mut model,
            )
        };

        if err != 0 {
            log::error!("dv_model_load_from_file error: {err}");
            return Err(err.into());
        }

        Ok(Model::new(Arc::clone(&self.session), self.ptr, model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_endpoints_nonempty() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().expect("should list endpoints");
        assert!(!endpoints.is_empty(), "should have at least one endpoint");
    }

    #[test]
    fn test_endpoint_status_valid_state() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let state = endpoint.check_status().expect("should get endpoint status");

        // State should be one of the valid variants (not a conversion error)
        match state {
            State::Init
            | State::Idle
            | State::Active
            | State::ActiveSlow
            | State::ActiveBoosted
            | State::ThermalInactive
            | State::ThermalUnknown
            | State::Inactive
            | State::Fault => {} // all valid
        }
    }

    #[test]
    fn test_dram_statistics_nonzero() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let stats = endpoint
            .dram_statistics()
            .expect("should get DRAM statistics");
        assert!(stats.dram_size > 0, "DRAM size should be non-zero");
    }
}
