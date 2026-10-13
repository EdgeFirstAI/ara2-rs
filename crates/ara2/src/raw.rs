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
}

/// # Safety
/// `model` must be a live model handle from the library `abi` describes.
pub(crate) unsafe fn model_fields(abi: Abi, model: *const dv_model) -> ModelFields {
    unsafe {
        with_layout!(abi, model => dv_model, |m| ModelFields {
            num_inputs: m.num_inputs as usize,
            num_outputs: m.num_outputs as usize,
            input_param: m.input_param,
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
        with_layout!(abi, stats => dv_endpoint_statistics, |s| EndpointStatsFields {
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
            },
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

    macro_rules! stats_roundtrip {
        ($name:ident, $abi:expr, $m:ident) => {
            #[test]
            fn $name() {
                use ara2_sys::$m::dv_endpoint_statistics;
                // SAFETY: all-zero is a valid bit pattern for this
                // plain-data struct.
                let mut s: dv_endpoint_statistics = unsafe { std::mem::zeroed() };
                s.ep_sys_clk = 1234;
                s.ep_dram_stats.ep_total_dram_size = 0x1_2345_6789;
                s.ep_dram_stats.ep_total_free_size = 77;
                let ptr: *const dv_endpoint_statistics = &s;
                let f = unsafe { endpoint_stats_fields($abi, ptr.cast()) };
                assert_eq!(f.sys_clk_mhz, 1234);
                assert_eq!(f.dram.total, 0x1_2345_6789);
                assert_eq!(f.dram.free, 77);
            }
        };
    }

    stats_roundtrip!(endpoint_stats_read_v1_1, Abi::V1_1, v1_1);
    stats_roundtrip!(endpoint_stats_read_v1_3, Abi::V1_3, v1_3);
    stats_roundtrip!(endpoint_stats_read_v3_0, Abi::V3_0, v3_0);
}
