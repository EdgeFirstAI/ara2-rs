#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
//! FFI bindings to `libaraclient`, the Kinara ARA-2 client library.
//!
//! The crate root holds what every supported DVAPI version shares. The
//! structs whose layout differs between versions are opaque here and
//! declared in full, at that version's layout, in [`v1_1`], [`v1_3`] and
//! [`v3_0`]; each of those modules also has a function table for the
//! functions that exist only in, or differ in signature in, that version.
//! The declarations live in `include/` and are checked against each SDK's
//! own headers by `tests/verify_sdk.rs`.

include!("ffi.rs");

/// DVAPI 1.1 (Kinara ARA-2 runtime 1.2.x).
pub mod v1_1 {
    include!("v1_1.rs");
}

/// DVAPI 1.3 (NXP rt-sdk-ara2 2.1.x).
pub mod v1_3 {
    include!("v1_3.rs");
}

/// DVAPI 3.0 (arart r3.0).
pub mod v3_0 {
    include!("v3_0.rs");
}

/// Asserts a struct's size and the offset of each named field.
///
/// The values were confirmed against each SDK by `tests/verify_sdk.rs`;
/// pinning them here makes an accidental edit to a declaration a build
/// error instead of a silent misread.
macro_rules! pin {
    ($ty:ty, $size:expr $(, $field:ident @ $off:expr)* $(,)?) => {
        const _: () = {
            assert!(::core::mem::size_of::<$ty>() == $size);
            $(assert!(::core::mem::offset_of!($ty, $field) == $off);)*
        };
    };
}

#[cfg(target_pointer_width = "64")]
mod pins {
    use super::*;

    pin!(v1_1::dv_model, 128, version @ 24, name @ 32, model_type @ 40, internal_name @ 48,
        num_inputs @ 56, num_outputs @ 60, input_param @ 72, output_param @ 80);
    pin!(v1_3::dv_model, 128, version @ 24, name @ 32, model_type @ 40, internal_name @ 48,
        num_inputs @ 56, num_outputs @ 60, input_param @ 72, output_param @ 80);
    pin!(v3_0::dv_model, 152, version @ 24, name @ 32, hw_cycles @ 40, model_type @ 48,
        internal_name @ 56, num_inputs @ 64, num_outputs @ 68, input_param @ 80,
        output_param @ 88, is_mcs @ 129, mcs_params @ 136);

    pin!(v1_1::dv_model_output_param, 112, postprocess_param @ 0, layer_id @ 8, blob_id @ 12,
        fused_parent_id @ 16, layer_name @ 24, blob_name @ 32, layer_fused_parent_name @ 40,
        layer_type @ 48, layout @ 56, size @ 64, width @ 68, height @ 72, depth @ 76, nch @ 80,
        bpp @ 84, num_classes @ 88, layer_output_type @ 92, num @ 96, max_dynamic_id @ 100);
    pin!(v1_3::dv_model_output_param, 120, postprocess_param @ 0, layer_id @ 8, blob_id @ 12,
        fused_parent_id @ 16, layer_name @ 24, blob_name @ 32, layer_fused_parent_name @ 40,
        layer_type @ 48, layout @ 56, size @ 64, width @ 68, height @ 72, depth @ 76, nch @ 80,
        bpp @ 84, num_classes @ 88, layer_output_type @ 92, num @ 96, max_dynamic_id @ 100,
        has_nms_parent @ 112);
    pin!(v3_0::dv_model_output_param, 120, postprocess_param @ 0, layer_id @ 8, blob_id @ 12,
        fused_parent_id @ 16, layer_name @ 24, blob_name @ 32, layer_fused_parent_name @ 40,
        layer_type @ 48, layout @ 56, size @ 64, width @ 68, height @ 72, depth @ 76, nch @ 80,
        bpp @ 84, num_classes @ 88, layer_output_type @ 92, num @ 96, max_dynamic_id @ 100,
        has_nms_parent @ 112);

    pin!(v1_1::dv_endpoint_dram_statistics, 56, ep_total_dram_size @ 8,
        ep_total_tensor_occupancy_size @ 48);
    pin!(v1_3::dv_endpoint_dram_statistics, 56, ep_total_dram_size @ 8,
        ep_total_tensor_occupancy_size @ 48);
    pin!(v3_0::dv_endpoint_dram_statistics, 64, ep_total_dram_size @ 8,
        ep_total_tensor_occupancy_size @ 48, ep_largest_contiguous_memory_size @ 56);

    pin!(v1_1::dv_endpoint_statistics, 120, state @ 8, ep_sys_clk @ 12, ep_dram_clk @ 16,
        ep_core_voltage @ 20, ep_temp @ 24, num_inference_queues @ 28, ep_infq_stats @ 32,
        num_active_models @ 40, model_stats @ 48, ep_dram_stats @ 56, ep_power_state @ 112,
        ep_soft_reset_count @ 116);
    pin!(v1_3::dv_endpoint_statistics, 128, state @ 8, ep_sys_clk @ 12, ep_dram_clk @ 16,
        ep_core_voltage @ 20, ep_temp @ 24, num_inference_queues @ 28, ep_infq_stats @ 32,
        num_active_models @ 40, model_stats @ 48, ep_dram_stats @ 56, ep_power_state @ 112,
        ep_soft_reset_count @ 116, ep_sbp_clk @ 120, ep_nnp_clk @ 124);
    pin!(v3_0::dv_endpoint_statistics, 136, state @ 8, ep_sys_clk @ 12, ep_dram_clk @ 16,
        ep_core_voltage @ 20, ep_temp @ 24, num_inference_queues @ 28, ep_infq_stats @ 32,
        num_active_models @ 40, model_stats @ 48, ep_dram_stats @ 56, ep_power_state @ 120,
        ep_soft_reset_count @ 124, ep_sbp_clk @ 128, ep_nnp_clk @ 132);

    pin!(v1_3::dv_status, 16, main_code @ 0, sub_code @ 4, type_ @ 8, criticality @ 12);
    pin!(v3_0::dv_status, 16, main_code @ 0, sub_code @ 4, type_ @ 8, criticality @ 12);
}

/// Type pins: each version's structs must point at that version's own
/// nested structs. Were a field declared through a common `_t` typedef it
/// would point at the zero-sized opaque type, and indexing it would stride
/// by zero. These functions are never called; they only have to compile.
#[allow(unused)]
mod type_pins {
    use super::*;

    fn v1_1_output(m: &v1_1::dv_model) -> *mut v1_1::dv_model_output_param {
        m.output_param
    }
    fn v1_3_output(m: &v1_3::dv_model) -> *mut v1_3::dv_model_output_param {
        m.output_param
    }
    fn v3_0_output(m: &v3_0::dv_model) -> *mut v3_0::dv_model_output_param {
        m.output_param
    }
    fn v1_1_dram(s: &v1_1::dv_endpoint_statistics) -> v1_1::dv_endpoint_dram_statistics {
        s.ep_dram_stats
    }
    fn v1_3_dram(s: &v1_3::dv_endpoint_statistics) -> v1_3::dv_endpoint_dram_statistics {
        s.ep_dram_stats
    }
    fn v3_0_dram(s: &v3_0::dv_endpoint_statistics) -> v3_0::dv_endpoint_dram_statistics {
        s.ep_dram_stats
    }
}
