use std::{
    ffi::{CString, c_char},
    path::Path,
    sync::Arc,
};

use ara2_sys::{
    DV_ENDPOINT_STATE, DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_ACTIVE as ACTIVE,
    DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_ACTIVE_SLOW as ACTIVE_SLOW,
    DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_FAULT as FAULT,
    DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_IDLE as IDLE,
    DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_INACTIVE as INACTIVE,
    DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_INIT as INIT,
    DV_ENDPOINT_STATE_DV_ENDPOINT_STATE_THERMAL_UNKNOWN as THERMAL_UNKNOWN,
    DV_MODEL_PRIORITY_LEVEL_DV_MODEL_PRIORITY_LEVEL_DEFAULT, dv_endpoint,
    dv_endpoint_dram_statistics, dv_endpoint_statistics, dv_model,
    v1_1::{
        dv_endpoint_state_1_1_DV_ENDPOINT_STATE_ACTIVE_BOOSTED as V11_ACTIVE_BOOSTED,
        dv_endpoint_state_1_1_DV_ENDPOINT_STATE_THERMAL_INACTIVE as V11_THERMAL_INACTIVE,
    },
    v1_3::{
        dv_endpoint_state_1_3_DV_ENDPOINT_STATE_FAIL_SAFE as V13_FAIL_SAFE,
        dv_endpoint_state_1_3_DV_ENDPOINT_STATE_THERMAL_ACTIVE_SLOW as V13_THERMAL_ACTIVE_SLOW,
    },
};

use crate::{
    Abi, Model,
    error::Error,
    session::{EndpointList, SessionInner},
};

/// The operational state of an NPU endpoint.
///
/// The variant set is the union of both DVAPI generations, because 1.3
/// reuses values 4 and 5 for names 1.1 does not have. Decoding therefore
/// takes the [`Abi`] of the library that produced the value, and a variant
/// documented as belonging to one generation is only ever produced under
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum State {
    /// Initializing; not yet ready to load models.
    Init,
    /// Ready to load models or run inference.
    Idle,
    /// Running inference at the nominal clock.
    Active,
    /// Running inference at a reduced clock to save power.
    ActiveSlow,
    /// DVAPI 1.1 value 4. Upstream documents it as "operating at reduced
    /// frequency", which its name contradicts; 1.3 renamed the same value
    /// [`State::ThermalActiveSlow`] and kept the documentation, so the
    /// reduced-clock reading is the likely one.
    ActiveBoosted,
    /// DVAPI 1.1 value 5, documented as thermally inactive. 1.3 renamed it
    /// [`State::FailSafe`] without updating that documentation, so whether
    /// the rename corrected the name or changed the meaning is unsettled.
    ThermalInactive,
    /// Running inference at a reduced clock because of thermal limits.
    /// DVAPI 1.3 value 4.
    ThermalActiveSlow,
    /// Operating in a restricted fail-safe mode. DVAPI 1.3 value 5.
    FailSafe,
    /// Thermal state cannot be determined.
    ThermalUnknown,
    /// Powered down or otherwise unavailable.
    Inactive,
    /// Unrecoverable hardware fault.
    Fault,
    /// A value neither generation defines, carried with the [`Abi`] it was
    /// read under. The number alone does not identify it, since the
    /// generations disagree about what some numbers mean.
    Unknown(DV_ENDPOINT_STATE, Abi),
}

impl State {
    /// Decodes a raw `dv_endpoint_state_t` against the DVAPI generation of
    /// the library that produced it.
    ///
    /// Matches each generation's own generated constants rather than bare
    /// integers. Upstream reuses these values, so a literal match compiles
    /// and runs unchanged while reporting the wrong state; going through
    /// the constants makes a future reassignment a build error.
    pub fn from_raw(value: DV_ENDPOINT_STATE, abi: Abi) -> Self {
        match (value, abi) {
            (INIT, _) => State::Init,
            (IDLE, _) => State::Idle,
            (ACTIVE, _) => State::Active,
            (ACTIVE_SLOW, _) => State::ActiveSlow,
            (THERMAL_UNKNOWN, _) => State::ThermalUnknown,
            (INACTIVE, _) => State::Inactive,
            (FAULT, _) => State::Fault,
            (V11_ACTIVE_BOOSTED, Abi::V1_1) => State::ActiveBoosted,
            (V11_THERMAL_INACTIVE, Abi::V1_1) => State::ThermalInactive,
            (V13_THERMAL_ACTIVE_SLOW, Abi::V1_3) => State::ThermalActiveSlow,
            (V13_FAIL_SAFE, Abi::V1_3) => State::FailSafe,
            _ => State::Unknown(value, abi),
        }
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

/// Live operational statistics for an endpoint.
///
/// Sourced from `dv_endpoint_get_statistics`. Note the Ara SDK exposes no
/// power-in-watts reading for the dNPU — temperature, core voltage, and
/// the clocks are the available power-related telemetry.
#[derive(Debug, Clone, Copy)]
pub struct EndpointStatistics {
    /// Current operational state of the endpoint.
    pub state: State,
    /// Endpoint system core clock, in MHz.
    pub sys_clk_mhz: i32,
    /// Endpoint DRAM clock, in MHz.
    pub dram_clk_mhz: i32,
    /// Average core voltage across the hardware measurement points, in volts.
    pub core_voltage_v: f32,
    /// Average endpoint temperature across the measurement points, in °C.
    pub temp_c: f32,
    /// DRAM usage statistics for the endpoint.
    pub dram: DramStatistics,
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

        Ok(State::from_raw(state, self.session.abi))
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

        // SAFETY: the call succeeded, so `dram_stats` points at the library's
        // statistics for this endpoint.
        let d = unsafe { crate::raw::dram_fields(self.session.abi, dram_stats) };
        let stats = DramStatistics {
            dram_size: d.total,
            dram_occupancy_size: d.occupied,
            free_size: d.free,
            reserved_occupancy_size: d.reserved,
            model_occupancy_size: d.models,
            tensor_occupancy_size: d.tensors,
        };

        unsafe {
            self.session
                .lib
                .dv_endpoint_free_dram_statistics(dram_stats, ep_count);
        }

        Ok(stats)
    }

    /// Read live operational statistics for this endpoint: clocks, core
    /// voltage, temperature, operational state, and DRAM usage.
    ///
    /// This is the dNPU's power-and-thermal telemetry. The Ara SDK does
    /// not expose a power-in-watts reading, so [`crate::EndpointStatistics`]
    /// surfaces the temperature and core voltage instead.
    pub fn statistics(&self) -> Result<EndpointStatistics, Error> {
        let mut ep_count = 1;
        let mut stats: *mut dv_endpoint_statistics = std::ptr::null_mut();
        let err = unsafe {
            self.session.lib.dv_endpoint_get_statistics(
                self.session.ptr,
                self.ptr,
                &mut stats,
                &mut ep_count,
            )
        };

        if err != 0 {
            return Err(err.into());
        }

        if stats.is_null() || ep_count <= 0 {
            return Err(Error::NullPointer(
                "dv_endpoint_get_statistics returned success with a null or empty buffer"
                    .to_owned(),
            ));
        }

        // Copy out the primitive fields before freeing the SDK buffer, so
        // a fallible `State` conversion can never leak the allocation.
        // SAFETY: checked above that `stats` is non-null and `ep_count` >= 1.
        let raw = unsafe { crate::raw::endpoint_stats_fields(self.session.abi, stats) };
        let dram = DramStatistics {
            dram_size: raw.dram.total,
            dram_occupancy_size: raw.dram.occupied,
            free_size: raw.dram.free,
            reserved_occupancy_size: raw.dram.reserved,
            model_occupancy_size: raw.dram.models,
            tensor_occupancy_size: raw.dram.tensors,
        };

        unsafe {
            self.session
                .lib
                .dv_endpoint_free_statistics(stats, ep_count);
        }

        Ok(EndpointStatistics {
            state: State::from_raw(raw.state, self.session.abi),
            sys_clk_mhz: raw.sys_clk_mhz,
            dram_clk_mhz: raw.dram_clk_mhz,
            core_voltage_v: raw.core_voltage_v,
            temp_c: raw.temp_c,
            dram,
        })
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

        // Decoding is infallible, so the check that carries weight is that
        // the value resolved to a named state rather than falling through
        // to Unknown -- which would mean the endpoint reported something
        // neither DVAPI generation defines.
        assert!(
            !matches!(state, State::Unknown(..)),
            "endpoint reported {state:?} under {}",
            session.abi()
        );
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

    #[test]
    fn test_statistics_nonzero() {
        let session = crate::tests::test_session();
        let endpoints = session.list_endpoints().unwrap();
        let endpoint = &endpoints[0];
        let stats = endpoint.statistics().expect("should get statistics");
        assert!(stats.sys_clk_mhz > 0, "system clock should be non-zero");
        assert!(stats.dram_clk_mhz > 0, "DRAM clock should be non-zero");
        assert!(stats.temp_c.is_finite(), "temperature should be finite");
        assert!(
            stats.core_voltage_v.is_finite(),
            "core voltage should be finite"
        );
        assert!(stats.dram.dram_size > 0, "DRAM size should be non-zero");
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;

    /// The two values DVAPI 1.3 reuses. A single mapping cannot serve both
    /// generations, and getting it wrong is silent: the integers are the
    /// same, so only the reported name differs.
    #[test]
    fn state_4_and_5_decode_per_generation() {
        assert_eq!(State::from_raw(4, Abi::V1_1), State::ActiveBoosted);
        assert_eq!(State::from_raw(5, Abi::V1_1), State::ThermalInactive);
        assert_eq!(State::from_raw(4, Abi::V1_3), State::ThermalActiveSlow);
        assert_eq!(State::from_raw(5, Abi::V1_3), State::FailSafe);
    }

    #[test]
    fn state_shared_values_decode_alike() {
        for (raw, expected) in [
            (0, State::Init),
            (1, State::Idle),
            (2, State::Active),
            (3, State::ActiveSlow),
            (6, State::ThermalUnknown),
            (7, State::Inactive),
            (8, State::Fault),
        ] {
            assert_eq!(State::from_raw(raw, Abi::V1_1), expected);
            assert_eq!(State::from_raw(raw, Abi::V1_3), expected);
        }
    }

    #[test]
    fn state_unknown_keeps_raw_value_and_generation() {
        assert_eq!(
            State::from_raw(1007, Abi::V1_3),
            State::Unknown(1007, Abi::V1_3)
        );
        assert_eq!(
            State::from_raw(1007, Abi::V1_1),
            State::Unknown(1007, Abi::V1_1)
        );
    }
}
