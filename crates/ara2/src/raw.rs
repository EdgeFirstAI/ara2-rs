//! Version-dispatched reads of `libaraclient` structs whose layout differs
//! between DVAPI versions.
//!
//! The common bindings declare these structs as opaque. [`with_layout!`]
//! casts an opaque pointer to the `ara2_sys` version module selected by the
//! session's [`Abi`] and evaluates its body once per arm, so every field
//! access is type-checked against that version's own layout. Nothing here
//! assumes one version's layout is a prefix of another's.
//!
//! This is the only module that may read a version-specific struct.

use std::ffi::c_char;

use ara2_sys::{
    dv_endpoint_dram_statistics, dv_endpoint_state_t, dv_endpoint_statistics,
    dv_layer_output_type_t, dv_model, dv_model_input_param, dv_model_output_postprocess_param,
    dv_version,
};

use crate::Abi;

/// Binds `$v` to `$ptr` cast to `ara2_sys::<version>::$ty` and evaluates
/// `$body` for the version `$abi` selects. Must be used in an `unsafe`
/// context; `$ptr` must point at a live struct the loaded library produced.
macro_rules! with_layout {
    ($abi:expr, $ptr:expr => $ty:ident, |$v:ident| $body:expr) => {{
        let _: *const ::ara2_sys::$ty = $ptr;
        match $abi {
            $crate::Abi::V1_1 => {
                let $v = &*($ptr as *const ::ara2_sys::v1_1::$ty);
                $body
            }
            $crate::Abi::V1_3 => {
                let $v = &*($ptr as *const ::ara2_sys::v1_3::$ty);
                $body
            }
            $crate::Abi::V3_0 => {
                let $v = &*($ptr as *const ::ara2_sys::v3_0::$ty);
                $body
            }
        }
    }};
}

/// The `dv_model` fields ara2 reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelFields {
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub input_param: *const dv_model_input_param,
    pub version: dv_version,
    pub name: *const c_char,
    pub model_type: u32,
}

/// The `dv_model_output_param` fields ara2 reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OutputFields {
    pub postprocess_param: *const dv_model_output_postprocess_param,
    pub layer_id: i32,
    pub blob_id: i32,
    pub fused_parent_id: i32,
    pub layer_name: *const c_char,
    pub blob_name: *const c_char,
    pub layer_fused_parent_name: *const c_char,
    pub layer_type: *const c_char,
    pub layout: *const c_char,
    pub size: i32,
    pub width: i32,
    pub height: i32,
    pub nch: i32,
    pub bpp: i32,
    pub num_classes: i32,
    pub layer_output_type: dv_layer_output_type_t,
    pub max_dynamic_id: i32,
}

/// The `dv_endpoint_dram_statistics` fields ara2 reads, in bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DramFields {
    pub total: u64,
    pub occupied: u64,
    pub free: u64,
    pub reserved: u64,
    pub models: u64,
    pub tensors: u64,
    pub largest_free: Option<u64>,
}

/// The `dv_endpoint_statistics` fields ara2 reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EndpointStatsFields {
    pub state: dv_endpoint_state_t,
    pub sys_clk_mhz: i32,
    pub dram_clk_mhz: i32,
    pub core_voltage_v: f32,
    pub temp_c: f32,
    pub dram: DramFields,
    pub soft_reset_count: u32,
    pub sbp_clk_mhz: Option<i32>,
    pub nnp_clk_mhz: Option<i32>,
}

/// Fields that exist only in some versions, read uniformly.
pub(crate) trait VersionFields {
    fn largest_free(&self) -> Option<u64> {
        None
    }
    fn sbp_nnp(&self) -> (Option<i32>, Option<i32>) {
        (None, None)
    }
}

impl VersionFields for ara2_sys::v1_1::dv_endpoint_dram_statistics {}
impl VersionFields for ara2_sys::v1_3::dv_endpoint_dram_statistics {}
impl VersionFields for ara2_sys::v3_0::dv_endpoint_dram_statistics {
    fn largest_free(&self) -> Option<u64> {
        Some(self.ep_largest_contiguous_memory_size)
    }
}
impl VersionFields for ara2_sys::v1_1::dv_endpoint_statistics {}
impl VersionFields for ara2_sys::v1_3::dv_endpoint_statistics {
    fn sbp_nnp(&self) -> (Option<i32>, Option<i32>) {
        (Some(self.ep_sbp_clk), Some(self.ep_nnp_clk))
    }
}
impl VersionFields for ara2_sys::v3_0::dv_endpoint_statistics {
    fn sbp_nnp(&self) -> (Option<i32>, Option<i32>) {
        (Some(self.ep_sbp_clk), Some(self.ep_nnp_clk))
    }
}

/// # Safety
/// `model` must be a live model handle from the library `abi` describes.
pub(crate) unsafe fn model_fields(abi: Abi, model: *const dv_model) -> ModelFields {
    unsafe {
        with_layout!(abi, model => dv_model, |m| ModelFields {
            num_inputs: m.num_inputs as usize,
            num_outputs: m.num_outputs as usize,
            input_param: m.input_param,
            version: m.version,
            name: m.name,
            model_type: m.model_type,
        })
    }
}

/// # Safety
/// `model` must be a live model handle from the library `abi` describes,
/// and `idx` less than its output count.
pub(crate) unsafe fn output_fields(abi: Abi, model: *const dv_model, idx: usize) -> OutputFields {
    unsafe {
        with_layout!(abi, model => dv_model, |m| {
            let o = &*m.output_param.add(idx);
            OutputFields {
                postprocess_param: o.postprocess_param,
                layer_id: o.layer_id,
                blob_id: o.blob_id,
                fused_parent_id: o.fused_parent_id,
                layer_name: o.layer_name,
                blob_name: o.blob_name,
                layer_fused_parent_name: o.layer_fused_parent_name,
                layer_type: o.layer_type,
                layout: o.layout,
                size: o.size,
                width: o.width,
                height: o.height,
                nch: o.nch,
                bpp: o.bpp,
                num_classes: o.num_classes,
                layer_output_type: o.layer_output_type,
                max_dynamic_id: o.max_dynamic_id,
            }
        })
    }
}

/// # Safety
/// `stats` must point at a DRAM statistics struct the library `abi`
/// describes returned.
pub(crate) unsafe fn dram_fields(
    abi: Abi,
    stats: *const dv_endpoint_dram_statistics,
) -> DramFields {
    unsafe {
        with_layout!(abi, stats => dv_endpoint_dram_statistics, |d| DramFields {
            total: d.ep_total_dram_size,
            occupied: d.ep_total_dram_occupancy_size,
            free: d.ep_total_free_size,
            reserved: d.ep_total_reserved_occupancy_size,
            models: d.ep_total_model_occupancy_size,
            tensors: d.ep_total_tensor_occupancy_size,
            largest_free: d.largest_free(),
        })
    }
}

/// # Safety
/// `stats` must point at an endpoint statistics struct the library `abi`
/// describes returned.
pub(crate) unsafe fn endpoint_stats_fields(
    abi: Abi,
    stats: *const dv_endpoint_statistics,
) -> EndpointStatsFields {
    unsafe {
        with_layout!(abi, stats => dv_endpoint_statistics, |s| {
            let (sbp_clk_mhz, nnp_clk_mhz) = s.sbp_nnp();
            EndpointStatsFields {
            state: s.state,
            sys_clk_mhz: s.ep_sys_clk,
            dram_clk_mhz: s.ep_dram_clk,
            core_voltage_v: s.ep_core_voltage,
            temp_c: s.ep_temp,
            dram: DramFields {
                total: s.ep_dram_stats.ep_total_dram_size,
                occupied: s.ep_dram_stats.ep_total_dram_occupancy_size,
                free: s.ep_dram_stats.ep_total_free_size,
                reserved: s.ep_dram_stats.ep_total_reserved_occupancy_size,
                models: s.ep_dram_stats.ep_total_model_occupancy_size,
                tensors: s.ep_dram_stats.ep_total_tensor_occupancy_size,
                largest_free: s.ep_dram_stats.largest_free(),
            },
            soft_reset_count: s.ep_soft_reset_count,
            sbp_clk_mhz,
            nnp_clk_mhz,
            }
        })
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;

    /// Builds a version's `dv_model` over a two-element output array and
    /// checks `model_fields` and `output_fields` read element 1 back.
    macro_rules! model_roundtrip {
        ($name:ident, $abi:expr, $m:ident) => {
            #[test]
            fn $name() {
                use ara2_sys::$m::{dv_model, dv_model_output_param};
                // SAFETY: all-zero is a valid bit pattern for these
                // plain-data structs (null pointers, zero integers).
                let mut outs: [dv_model_output_param; 2] = unsafe { std::mem::zeroed() };
                outs[0].layer_id = 10;
                outs[0].width = 100;
                outs[1].layer_id = 11;
                outs[1].width = 111;
                outs[1].height = 7;
                let mut model: dv_model = unsafe { std::mem::zeroed() };
                model.num_inputs = 3;
                model.num_outputs = 2;
                model.output_param = outs.as_mut_ptr();
                let ptr: *const dv_model = &model;

                let m = unsafe { model_fields($abi, ptr.cast()) };
                assert_eq!((m.num_inputs, m.num_outputs), (3, 2));
                let o = unsafe { output_fields($abi, ptr.cast(), 1) };
                assert_eq!((o.layer_id, o.width, o.height), (11, 111, 7));
            }
        };
    }

    model_roundtrip!(model_reads_v1_1, Abi::V1_1, v1_1);
    model_roundtrip!(model_reads_v1_3, Abi::V1_3, v1_3);
    model_roundtrip!(model_reads_v3_0, Abi::V3_0, v3_0);

    /// `$extra` sets the fields only some versions have: SBP 355 / NNP 450
    /// where `$sbp`, largest free block 4096 where `$largest` is `Some`.
    macro_rules! stats_roundtrip {
        ($name:ident, $abi:expr, $m:ident, sbp: $sbp:expr, largest: $largest:expr, $extra:expr) => {
            #[test]
            fn $name() {
                use ara2_sys::$m::dv_endpoint_statistics;
                // SAFETY: all-zero is a valid bit pattern for this
                // plain-data struct.
                let mut s: [dv_endpoint_statistics; 2] = unsafe { std::mem::zeroed() };
                s[0].ep_sys_clk = 1;
                s[0].ep_dram_stats.ep_total_dram_size = 2;
                s[0].ep_dram_stats.ep_total_free_size = 3;
                s[1].ep_sys_clk = 1234;
                s[1].ep_dram_clk = 1066;
                s[1].ep_soft_reset_count = 5;
                s[1].ep_dram_stats.ep_total_dram_size = 0x1_2345_6789;
                s[1].ep_dram_stats.ep_total_free_size = 77;
                let extra: fn(&mut dv_endpoint_statistics) = $extra;
                extra(&mut s[1]);
                // Element 1 sits one version-specific stride in, so an arm
                // wired to another version's layout reads the wrong bytes.
                let ptr: *const dv_endpoint_statistics = &s[1];
                let f = unsafe { endpoint_stats_fields($abi, ptr.cast()) };
                assert_eq!(f.sys_clk_mhz, 1234);
                assert_eq!(f.dram_clk_mhz, 1066);
                assert_eq!(f.dram.total, 0x1_2345_6789);
                assert_eq!(f.dram.free, 77);
                assert_eq!(f.soft_reset_count, 5);
                let (sbp, nnp) = if $sbp {
                    (Some(355), Some(450))
                } else {
                    (None, None)
                };
                assert_eq!((f.sbp_clk_mhz, f.nnp_clk_mhz), (sbp, nnp));
                assert_eq!(f.dram.largest_free, $largest);
            }
        };
    }

    stats_roundtrip!(endpoint_stats_read_v1_1, Abi::V1_1, v1_1, sbp: false, largest: None, |_s| {});
    stats_roundtrip!(endpoint_stats_read_v1_3, Abi::V1_3, v1_3, sbp: true, largest: None, |s| {
        s.ep_sbp_clk = 355;
        s.ep_nnp_clk = 450;
    });
    stats_roundtrip!(endpoint_stats_read_v3_0, Abi::V3_0, v3_0, sbp: true, largest: Some(4096), |s| {
        s.ep_sbp_clk = 355;
        s.ep_nnp_clk = 450;
        s.ep_dram_stats.ep_largest_contiguous_memory_size = 4096;
    });
}
