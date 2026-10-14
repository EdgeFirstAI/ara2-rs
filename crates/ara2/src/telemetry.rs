//! Telemetry the ARA-2 runtime reports: per-inference hardware counters,
//! endpoint statistics, and static endpoint and model information.

use std::time::Duration;

use ara2_sys::{
    DV_ENDPOINT_HOST_INTERFACE_DV_ENDPOINT_HOST_INTERFACE_PCIE,
    DV_ENDPOINT_HOST_INTERFACE_DV_ENDPOINT_HOST_INTERFACE_USB, dv_endpoint_info,
    dv_infer_statistics, timespec,
};

use crate::{Abi, ModelTiming};

/// Clock frequencies, in MHz. The NNP and SBP clocks are `None` where the
/// loaded DVAPI version does not report them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Clocks {
    /// System core clock.
    pub sys_mhz: i32,
    /// DRAM clock.
    pub dram_mhz: i32,
    /// Neural processor clock.
    pub nnp_mhz: Option<i32>,
    /// SBP clock.
    pub sbp_mhz: Option<i32>,
}

/// A point on the host's `CLOCK_MONOTONIC`. Inference timestamps are `Some`
/// on DVAPI 3.0, where the runtime stamps on that clock. Compare against
/// [`monotonic_now`] or any other `CLOCK_MONOTONIC` reading, such as a
/// Perfetto trace's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicTime(Duration);

impl MonotonicTime {
    /// Time since the `CLOCK_MONOTONIC` epoch.
    pub fn as_duration(self) -> Duration {
        self.0
    }

    /// The stamp as a monotonic time. The DVAPI 1.x runtimes stamp with the
    /// wall clock (`CLOCK_REALTIME`), so only versions known to use
    /// `CLOCK_MONOTONIC` are converted. An unset stamp is `None` too.
    fn from_timespec(t: timespec, abi: Abi) -> Option<Self> {
        let monotonic = matches!(abi, Abi::V3_0);
        (monotonic && (t.tv_sec > 0 || t.tv_nsec > 0)).then(|| {
            MonotonicTime(Duration::new(
                t.tv_sec.max(0) as u64,
                t.tv_nsec.clamp(0, 999_999_999) as u32,
            ))
        })
    }
}

/// The current `CLOCK_MONOTONIC` time.
pub fn monotonic_now() -> MonotonicTime {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `t` is a valid out-pointer; CLOCK_MONOTONIC is always available on Linux.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    MonotonicTime(Duration::new(t.tv_sec as u64, t.tv_nsec as u32))
}

/// Inference retries the proxy performed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Retries {
    /// Retries across this proxy session.
    pub session_total: u32,
    /// Retries of this inference.
    pub this_inference: u32,
}

/// Everything the runtime reports about one completed inference.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct InferenceStats {
    /// Execution and transfer durations.
    pub timing: ModelTiming,
    /// Clocks the inference ran at.
    pub clocks: Clocks,
    /// Hardware cycles, including floating-point work.
    pub hw_cycles: u32,
    /// Hardware cycles spent on floating-point work.
    pub fp_cycles: u32,
    /// When the input transfer started. `None` where the runtime does not
    /// stamp on `CLOCK_MONOTONIC` (DVAPI 1.x stamps with the wall clock).
    pub input_start: Option<MonotonicTime>,
    /// When the inference entered the NNP queue. `None` as for `input_start`.
    pub inference_start: Option<MonotonicTime>,
    /// When the output transfer started. `None` as for `input_start`.
    pub output_start: Option<MonotonicTime>,
    /// Retries the proxy performed.
    pub retries: Retries,
}

/// A runtime duration in microseconds as a [`Duration`], keeping sub-µs
/// precision. Negative, NaN and infinite values, which the runtime uses
/// for "not measured", become zero.
pub(crate) fn micros(v: f32) -> Duration {
    if v.is_finite() && v > 0.0 {
        Duration::from_nanos((f64::from(v) * 1000.0).round() as u64)
    } else {
        Duration::ZERO
    }
}

pub(crate) fn mhz(v: i32) -> Option<i32> {
    (v > 0).then_some(v)
}

pub(crate) fn inference_stats(raw: &dv_infer_statistics, abi: Abi) -> InferenceStats {
    InferenceStats {
        timing: ModelTiming {
            run_time: micros(raw.inference_execution_time),
            input_time: micros(raw.input_transfer_time),
            output_time: micros(raw.output_transfer_time),
        },
        clocks: Clocks {
            sys_mhz: raw.ep_hw_sys_clk,
            dram_mhz: raw.ep_hw_dram_clk,
            nnp_mhz: mhz(raw.ep_hw_nnp_clk),
            sbp_mhz: mhz(raw.ep_hw_sbp_clk),
        },
        hw_cycles: raw.ep_hw_total_inference_cycles,
        fp_cycles: raw.ep_hw_fp_cycles,
        input_start: MonotonicTime::from_timespec(raw.input_transfer_start_time_stamp, abi),
        inference_start: MonotonicTime::from_timespec(raw.inference_start_time_stamp, abi),
        output_start: MonotonicTime::from_timespec(raw.output_transfer_start_time_stamp, abi),
        retries: Retries {
            session_total: raw.cumulative_replay_count,
            this_inference: raw.current_replay_count,
        },
    }
}

/// Verbosity of the client library's own logging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    /// Everything.
    Trace = 0,
    /// Debug detail.
    Debug = 1,
    /// Informational messages.
    Info = 2,
    /// Warnings and worse.
    Warn = 3,
    /// Errors and worse.
    Error = 4,
    /// Critical errors only.
    Critical = 5,
    /// No logging.
    Off = 6,
}

/// A library-owned C string, trimmed; `None` for null or blank.
pub(crate) fn c_string(ptr: *const std::ffi::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the library returns NUL-terminated strings it keeps alive
    // for the lifetime of the owning struct.
    let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy();
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_owned())
}

/// Static information about the endpoint's hardware, read once from the
/// endpoint list.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EndpointInfo {
    /// PCI-style device identifier.
    pub device_id: u32,
    /// Vendor identifier.
    pub vendor_id: u32,
    /// The ARA-2 chip, when the runtime reports it.
    pub chip: Option<ChipInfo>,
    /// The attached DRAM, when the runtime reports it.
    pub dram: Option<DramInfo>,
    /// How the endpoint is attached to the host, when the runtime reports it.
    pub interface: Option<Interface>,
}

/// The ARA-2 chip an endpoint is built on.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ChipInfo {
    /// Chip name, such as `ARA-2`; `None` when the library reports none.
    pub id: Option<String>,
    /// Silicon revision; `None` when the library reports none.
    pub revision: Option<String>,
    /// Number of control processors.
    pub control_processors: i32,
    /// Number of neural processors.
    pub neural_processors: i32,
    /// L2 memory size in bytes.
    pub l2_bytes: u32,
}

/// The DRAM attached to an endpoint.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DramInfo {
    /// Manufacturer name, when the runtime knows it.
    pub vendor_name: Option<String>,
}

/// The host bus an endpoint is attached through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InterfaceKind {
    /// PCI Express.
    Pcie,
    /// USB.
    Usb,
    /// A host interface value this build does not recognise.
    Other(u32),
}

/// Where an endpoint sits on the host.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Interface {
    /// The host bus type.
    pub kind: InterfaceKind,
    /// Bus number.
    pub bus: i32,
    /// Device number.
    pub device: i32,
    /// Port number.
    pub port: i32,
    /// The device's sysfs directory; reported for PCIe only.
    pub sysfs_path: Option<std::path::PathBuf>,
}

pub(crate) fn endpoint_info(info: &dv_endpoint_info) -> EndpointInfo {
    // SAFETY: the sub-structs are owned by the endpoint list, which the
    // caller keeps alive, and a non-null pointer refers to a valid struct.
    let chip = unsafe { info.chip.as_ref() }.map(|c| ChipInfo {
        id: c_string(c.id),
        revision: c_string(c.rev),
        control_processors: c.control_processor_count,
        neural_processors: c.neural_processor_count,
        l2_bytes: c.l2_memory_size,
    });
    let dram = unsafe { info.dram.as_ref() }.map(|d| DramInfo {
        vendor_name: c_string(d.vendor_name),
    });
    let interface = unsafe { info.iface.as_ref() }.map(|i| {
        let kind = match i.type_ {
            DV_ENDPOINT_HOST_INTERFACE_DV_ENDPOINT_HOST_INTERFACE_PCIE => InterfaceKind::Pcie,
            DV_ENDPOINT_HOST_INTERFACE_DV_ENDPOINT_HOST_INTERFACE_USB => InterfaceKind::Usb,
            other => InterfaceKind::Other(other),
        };
        // The union has a single member, meaningful only for PCIe.
        let sysfs_path = (kind == InterfaceKind::Pcie)
            .then(|| c_string(unsafe { i.sysfs_path.pcie_dir }))
            .flatten()
            .map(std::path::PathBuf::from);
        Interface {
            kind,
            bus: i.bus_num,
            device: i.device_num,
            port: i.port_num,
            sysfs_path,
        }
    });
    EndpointInfo {
        device_id: info.device_id,
        vendor_id: info.vendor_id,
        chip,
        dram,
        interface,
    }
}

#[cfg(test)]
mod offline_tests {
    #[test]
    fn log_levels_match_the_header() {
        assert_eq!(LogLevel::Trace as u32, 0);
        assert_eq!(LogLevel::Warn as u32, 3);
        assert_eq!(LogLevel::Off as u32, 6);
    }

    use super::*;
    use crate::Abi;
    use ara2_sys::{dv_infer_statistics, timespec};

    fn ts(sec: i64, nsec: i64) -> timespec {
        timespec {
            tv_sec: sec as _,
            tv_nsec: nsec as _,
        }
    }

    fn raw() -> dv_infer_statistics {
        dv_infer_statistics {
            ep_hw_sys_clk: 550,
            ep_hw_nnp_clk: 450,
            ep_hw_sbp_clk: 355,
            ep_hw_dram_clk: 1066,
            ep_hw_total_inference_cycles: 4_912_063,
            ep_hw_fp_cycles: 0,
            input_transfer_time: 1928.0,
            output_transfer_time: 2362.5,
            ep_queue_submission_time: -1.0,
            cumulative_replay_count: 3,
            current_replay_count: 1,
            input_transfer_start_time_stamp: ts(498_204, 957_159_978),
            output_transfer_start_time_stamp: ts(498_204, 973_634_716),
            inference_start_time_stamp: ts(498_204, 959_091_866),
            inference_execution_time: 13836.8,
            input_ddr_address: 0,
            output_ddr_address: 0,
        }
    }

    #[test]
    fn c_string_trims_and_handles_null() {
        assert_eq!(c_string(std::ptr::null()), None);
        assert_eq!(c_string(c"ARA-2 ".as_ptr()), Some("ARA-2".to_owned()));
        assert_eq!(c_string(c"  ".as_ptr()), None);
    }

    #[test]
    fn micros_keeps_sub_microsecond_precision() {
        assert_eq!(micros(2362.5), Duration::from_nanos(2_362_500));
    }

    #[test]
    fn micros_clamps_sentinels() {
        assert_eq!(micros(-1.0), Duration::ZERO);
        assert_eq!(micros(f32::NAN), Duration::ZERO);
        assert_eq!(micros(f32::INFINITY), Duration::ZERO);
    }

    #[test]
    fn inference_stats_maps_every_field() {
        let s = inference_stats(&raw(), Abi::V3_0);
        assert_eq!(s.timing.output_time, Duration::from_nanos(2_362_500));
        assert_eq!(
            s.clocks,
            Clocks {
                sys_mhz: 550,
                dram_mhz: 1066,
                nnp_mhz: Some(450),
                sbp_mhz: Some(355)
            }
        );
        assert_eq!(s.hw_cycles, 4_912_063);
        assert_eq!(
            s.retries,
            Retries {
                session_total: 3,
                this_inference: 1
            }
        );
        let (input, inference, output) = (
            s.input_start.unwrap(),
            s.inference_start.unwrap(),
            s.output_start.unwrap(),
        );
        assert!(input < inference && inference < output);
        assert_eq!(input.as_duration(), Duration::new(498_204, 957_159_978));
    }

    #[test]
    fn unpopulated_clocks_are_none() {
        let mut r = raw();
        r.ep_hw_nnp_clk = 0;
        r.ep_hw_sbp_clk = 0;
        let s = inference_stats(&r, Abi::V3_0);
        assert_eq!((s.clocks.nnp_mhz, s.clocks.sbp_mhz), (None, None));
    }

    #[test]
    fn wall_clock_stamps_are_not_reported_as_monotonic() {
        for abi in [Abi::V1_1, Abi::V1_3] {
            let s = inference_stats(&raw(), abi);
            assert_eq!(
                (s.input_start, s.inference_start, s.output_start),
                (None, None, None)
            );
        }
    }

    #[test]
    fn unset_stamps_are_none() {
        let mut r = raw();
        r.inference_start_time_stamp = ts(0, 0);
        let s = inference_stats(&r, Abi::V3_0);
        assert_eq!(s.inference_start, None);
        assert!(s.input_start.is_some() && s.output_start.is_some());
    }
}
