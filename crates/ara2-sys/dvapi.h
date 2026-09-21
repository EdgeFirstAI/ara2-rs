/*
 * Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
 * Copyright 2025-2026 NXP
 *
 * NXP Proprietary. This software is owned or controlled by NXP and may only be
 * used strictly in accordance with the applicable license terms. By expressly
 * accepting such terms or by downloading, installing, activating and/or
 * otherwise using the software, you are agreeing that you have read, and that
 * you agree to comply with and are bound by, such license terms. If you do not
 * agree to be bound by the applicable license terms, then you may not retain,
 * install, activate or otherwise use the software.
 *
 */

/**
 * @file dvapi.h
 * @brief Public API for the Client library.
 *
 * Provides data structures and APIs for:
 * - Session management
 * - Endpoint discovery and statistics
 * - Model loading and management
 * - Inference execution (sync/async)
 * - Shared memory handling
 * - LLM configuration and control
 */

#ifndef __DV_API_H__
#define __DV_API_H__

#ifdef WIN32
#define EXPORT __declspec(dllexport)
#include <time.h>
#else
#define EXPORT
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
/// Header file to denote error types
#include "dv_status_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for an inference request.
 *
 * Forward-declared structure used as a type-safe handle to an in-flight
 * or completed inference request. Managed exclusively by the client library.
 */
struct dv_infer_request_handle;
typedef struct dv_infer_request_handle dv_infer_request_handle_t;

/**
 * @brief Opaque handle for a loaded model.
 *
 * Forward-declared structure used as a type-safe handle to a model loaded
 * onto an endpoint. Managed exclusively by the client library.
 */
struct dv_model_handle;
typedef struct dv_model_handle dv_model_handle_t;
 
/// DV session socket type
typedef enum DV_SESSION_SOCKET_TYPE {
  DV_SESSION_SOCKET_TYPE_UNIX = 0,    /**< unix domain socket */
  DV_SESSION_SOCKET_TYPE_TCPIPv4 = 1, /**< tcp ipv4 socket */
} dv_session_socket_type_t;

/// DV Endpoint to host communication interface
typedef enum DV_ENDPOINT_HOST_INTERFACE {
  DV_ENDPOINT_HOST_INTERFACE_PCIE = 1, /**< host and dv connected via pcie interface */
  DV_ENDPOINT_HOST_INTERFACE_USB = 2   /**< host and dv connected via usb interface */
} dv_endpoint_host_interface_t;

/**
 * @brief Represents predefined endpoint groupings available from the proxy server.
 */
// Endpoint default groups
typedef enum DV_ENDPOINT_DEFAULT_GROUP {
  DV_ENDPOINT_DEFAULT_GROUP_ALL = 0,  /**< default group for all the endpoint(s)
                                         connected to inference proxy server */
  DV_ENDPOINT_DEFAULT_GROUP_PCIE = 1, /**< default group for all the pcie endpoint(s) connected to inference
                                         proxy server */
  DV_ENDPOINT_DEFAULT_GROUP_USB = 2,  /**< default group for all the usb endpoint(s) connected to inference
                                         proxy server */
} dv_endpoint_default_group_t;

/// Endpoint state
typedef enum DV_ENDPOINT_STATE {
  DV_ENDPOINT_STATE_INIT = 0,                /**< endpoint is in init state */
  DV_ENDPOINT_STATE_IDLE = 1,                /**< endpoint is in idle state */
  DV_ENDPOINT_STATE_ACTIVE = 2,              /**< endpoint is in active state */
  DV_ENDPOINT_STATE_ACTIVE_SLOW = 3,         /**< endpoint is operating at reduced frequency*/
  DV_ENDPOINT_STATE_THERMAL_ACTIVE_SLOW = 4, /**< endpoint is operating at reduced frequency*/
  DV_ENDPOINT_STATE_FAIL_SAFE = 5,           /**< endpoint is in thermal Inactive state */
  DV_ENDPOINT_STATE_THERMAL_UNKNOWN = 6,     /**< endpoint is in unknown thermal state*/
  DV_ENDPOINT_STATE_INACTIVE = 7,            /**< endpoint is in Inactive state */
  DV_ENDPOINT_STATE_FAULT = 8,               /**< endpoint is in faulty state */
  DV_ENDPOINT_STATE_BAD_INTERFACE = 1001,    /**< [unsupported] */
  DV_ENDPOINT_STATE_RECOVERY = 1003,         /**< [unsupported] */
  DV_ENDPOINT_STATE_DEAD = 1004,             /**< [unsupported] */
  DV_ENDPOINT_STATE_DRAIN = 1005,            /**< [unsupported] */
  DV_ENDPOINT_STATE_POWER_GATED = 1006,      /**< [unsupported] */
  DV_ENDPOINT_STATE_CLOSED = 1007,           /**< [unsupported] */
} dv_endpoint_state_t;

/// Endpoint Power State [currently unsupported]
typedef enum DV_ENDPOINT_POWER_STATE {
  DV_POWER_STATE_L0 = 0,  /**< endpoint is in L0 state */
  DV_POWER_STATE_L1 = 1,  /**< endpoint is in L1 state */
  DV_POWER_STATE_L1A = 2, /**< endpoint is in L1A state*/
  DV_POWER_STATE_L2 = 3,  /**< endpoint is in L2 state */
} dv_endpoint_power_state_t;

/**
 * @brief Represents the group membership type of an endpoint collection.
 */
// Endpoint group type
typedef enum DV_ENDPOINT_GROUP_TYPE {
  DV_ENDPOINT_GROUP_TYPE_NONE = 0,   /**< endpoint group type none */
  DV_ENDPOINT_GROUP_TYPE_ALL = 1,    /**< endpoint group type all */
  DV_ENDPOINT_GROUP_TYPE_PCIE = 2,   /**< endpoint group type pcie */
  DV_ENDPOINT_GROUP_TYPE_USB = 3,    /**< endpoint group type usb */
  DV_ENDPOINT_GROUP_TYPE_CUSTOM = 4, /**< endpoint group type custom */
} dv_endpoint_group_type_t;

/// Model Network type
typedef enum DV_LAYER_OUTPUT_TYPE {
  DV_LAYER_OUTPUT_TYPE_CLASSIFICATION = 0,        /**< represents classification type of network */
  DV_LAYER_OUTPUT_TYPE_DETECTION = 1,             /**< represents detection type of network */
  DV_LAYER_OUTPUT_TYPE_SEMANTIC_SEGMENTATION = 2, /**< represents semantic segmentation type of network */
  DV_LAYER_OUTPUT_TYPE_RAW = 3,                   /**< represents all other network types which can't be determined */
} dv_layer_output_type_t;

/// Priority level of model
typedef enum DV_MODEL_PRIORITY_LEVEL {
  DV_MODEL_PRIORITY_LEVEL_LOW = 0,                                  /**< model priority low */
  DV_MODEL_PRIORITY_LEVEL_MEDIUM = 1,                               /**< model priority medium */
  DV_MODEL_PRIORITY_LEVEL_DEFAULT = DV_MODEL_PRIORITY_LEVEL_MEDIUM, /**< model priority default */
  DV_MODEL_PRIORITY_LEVEL_HIGH = 2,                                 /**< model priority high */
} dv_model_priority_level_t;

/// Inference status
typedef enum DV_INFERENCE_STATUS {
  DV_INFERENCE_STATUS_QUEUED = 0,    /**< Inference is in queued state */
  DV_INFERENCE_STATUS_RUNNING = 1,   /**< Inference is in running/executing state */
  DV_INFERENCE_STATUS_COMPLETED = 2, /**< Inference is in completed state */
  DV_INFERENCE_STATUS_FAILED = 4,    /**< Inference is in failed state */
  DV_INFERENCE_STATUS_UNKNOWN = 5,   /**< Inference information is not available */
} dv_inference_status_t;

/// Network type of model

/// Blob types
typedef enum DV_BLOB_TYPE {
  DV_BLOB_TYPE_RAW_POINTER = 0,    /**< represents blob backed by raw pointer */
  DV_BLOB_TYPE_SHM_DESCRIPTOR = 1, /**< represents blob backed by registered shared memory descriptor */
  DV_BLOB_TYPE_FD = 2,             /**< represents blob backed by non-registered file descriptor */
} dv_blob_type_t;

// changes for versioning

/**
 * @brief Represents the type of a versioned software or firmware product component.
 */
typedef enum DV_PRODUCT_TYPE {
  INVALID_PRODUCT = -1, /**< invalid product type */
  PROXY = 0,            /**< product is proxy */       
  PCI_DRIVER = 1,       /**< NOT USED CURRENTLY */
  FIRMWARE = 2,         /**< product is firmware */
  CNN_MODEL = 3,        /**< product is CNN model */
  LLM_MODEL = 4,        /**< product is LLM model */
  CLIENT_LIB = 5,       /**< product is client library */
  SYSAPI = 6,           /**< NOT USED CURRENTLY */
} dv_product_type_t;

// DV client log level to dump logs
typedef enum DV_CLIENT_LOG_LEVEL {
  DV_CLIENT_LOG_LEVEL_TRACE = 0,    /**< dump critical, error, warning, info, debug and trace logs */
  DV_CLIENT_LOG_LEVEL_DEBUG = 1,    /**< dump critical, error, warning, info and debug logs */
  DV_CLIENT_LOG_LEVEL_INFO = 2,     /**< dump critical, error, warning and info logs */
  DV_CLIENT_LOG_LEVEL_WARN = 3,     /**< dump critical, error and warning logs */
  DV_CLIENT_LOG_LEVEL_ERROR = 4,    /**< dump critical and error logs */
  DV_CLIENT_LOG_LEVEL_CRITICAL = 5, /**< dump critical logs */
  DV_CLIENT_LOG_LEVEL_OFF = 6,      /**< turn off client logs */
} dv_client_log_level_t;

/**
 * @brief Represents a semantic version number with four components.
 *
 * @var dv_version_t::major
 *   Major version number.
 * @var dv_version_t::minor
 *   Minor version number.
 * @var dv_version_t::patch
 *   Patch version number.
 * @var dv_version_t::patch_minor
 *   Sub-patch (minor patch) version number.
 */
typedef struct dv_version {
  uint8_t major;
  uint8_t minor;
  uint8_t patch;
  uint8_t patch_minor;
} dv_version_t;

/**
 * @brief Represents a product type paired with its version information.
 *
 * @var dv_product_version_t::product_type
 *   The type of the product component as defined by @ref DV_PRODUCT_TYPE.
 * @var dv_product_version_t::product_version
 *   The version of the product component.
 */
typedef struct dv_product_version {
  dv_product_type_t product_type;
  dv_version_t product_version;
} dv_product_version_t;

/**
 * @brief Represents a contiguous memory region used as input or output for inference and model operations.
 *
 * Use @ref DV_BLOB_TYPE_RAW_POINTER for host-allocated buffers passed directly to the API.
 * Use @ref DV_BLOB_TYPE_SHM_DESCRIPTOR when the buffer has been registered via @ref dv_shmfd_register
 * to avoid redundant host-to-device copies across inference requests.
 * Use @ref DV_BLOB_TYPE_FD for file-descriptor-backed buffers that have not been pre-registered.
 */
// DV blob
typedef struct dv_blob {
  void *handle;             /**< blob handle (raw pointer or shared file descriptor returned
                               by server) */
  uint64_t offset;          /**< blob offset */
  uint64_t size;            /**< blob size */
  dv_blob_type_t blob_type; /**< blob type as represented in enum DV_BLOB_TYPE */
} dv_blob_t;

/// Default session wide parameters that are to be passed to dv_session_create_via_*_with_options
typedef struct dv_session_options {
  int timeout_ms; /**< global default timeout */
} dv_session_options_t;

/// Session object with parameters
typedef struct dv_session {
  void *handle;                         /**< session private handle, managed by client library */
  char *socket_str;                     /**< NULL terminated socket connection string */
  dv_session_socket_type_t socket_type; /**< socket types: Unix domain socket/TCPIPv4 */
} dv_session_t;

/// Shared memory decriptor generated after registering fd to server
typedef struct dv_shm_descriptor {
  void *handle;          /**< shared mem handle, managed by client library */
  dv_session_t *session; /**< session handle on which shared memory is registered */
  int fd;                /**< file fd for which shared memory is registered */
  size_t size;           /**< size to map */
  size_t offset;         /**< offset in file */
  int fd_type;           /**< type of fd shared (reserved) */
} dv_shm_descriptor_t;

/**
 * @brief Represents the on-chip processor and memory configuration of a DV endpoint.
 */
// DV Endpoint chip information
typedef struct dv_endpoint_chip_info {
  char *id;                    /**< dv chip id */
  char *rev;                   /**< dv chip revision */
  int control_processor_count; /**< dv chip control processor count */
  int neural_processor_count;  /**< dv chip neural processor count */
  uint32_t l2_memory_size;     /**< dv chip internal L2 memory size in bytes */
} dv_endpoint_chip_info_t;

/**
 * @brief Represents the external DRAM device information for a DV endpoint.
 */
// DV Endpoint external dram information
typedef struct dv_endpoint_dram_info {
  uint32_t vendor_id; /**< dv dram vendor id */
  char *vendor_name;  /**< dv dram vendor name */
  uint32_t size;      /**< dv dram memory size in bytes */
  uint8_t rev_id1;    /**< dv dram revision id 1 */
  uint8_t rev_id2;    /**< dv dram revision id 2 */
  uint8_t density;    /**< dv dram density */
  uint8_t io_width;   /**< dv dram io width */
} dv_endpoint_dram_info_t;

/// DV Endpoint interface information
typedef struct dv_endpoint_iface_info {
  dv_endpoint_host_interface_t type; /**< dv module physical interface (pcie, usb) with host */
  int bus_num;                       /**< host interface bus number on which dv device is connected */
  int device_num;                    /**< host interface device number on which dv device is
                                        connected */
  union {
    char *pcie_dir; /**< path to the pcie sysfs entry for a PCIE
                       interface device */
  } sysfs_path;

  int port_num; /**< port on which device is connected */
} dv_endpoint_iface_info_t;

/// DV Endpoint information
typedef struct dv_endpoint_info {
  uint32_t device_id;              /**< endpoint device id */
  uint32_t vendor_id;              /**< endpoint vendor id */
  dv_endpoint_chip_info_t *chip;   /**< endpoint chip information */
  dv_endpoint_dram_info_t *dram;   /**< endpoint external dram information */
  dv_endpoint_iface_info_t *iface; /**< endpoint interface information */
  char *module_name;               /**< physical module name connected to server */
  uint32_t gpio0;                  /**< [unused]  */
  uint32_t gpio1;                  /**< [unused]  */
  uint32_t device_uid;             /**< used to uniquely identify the chip */
} dv_endpoint_info_t;

/// Endpoint/Endpoint Group object
typedef struct dv_endpoint {
  void *handle;                      /**< endpoint private handle, managed by client library */
  dv_session_t *session;             /**< session handle on which endpoint is queried */
  int num_ep;                        /**< number of endpoints in the group */
  dv_endpoint_group_type_t grp_type; /**< endpoint group type */
  dv_endpoint_info_t **ep_info_list; /**< list of configuration for all the
                                        endpoint(s) in the group */
} dv_endpoint_t;

/**
 * @brief Represents a DRAM utilization snapshot for a single endpoint.
 *
 * Use this to monitor endpoint memory pressure before loading additional models.
 * If @c ep_total_free_size is low, unload unused models before attempting new
 * loads to avoid out-of-memory errors during inference or model load.
 */
// Endpoint dram statistics
typedef struct dv_endpoint_dram_statistics {
  dv_endpoint_t *ep;                         /**< endpoint handle */
  uint64_t ep_total_dram_size;               /**< endpoint dram size in bytes */
  uint64_t ep_total_dram_occupancy_size;     /**< endpoint dram memory occupied in bytes */
  uint64_t ep_total_free_size;               /**< endpoint dram memory free in bytes */
  uint64_t ep_total_reserved_occupancy_size; /**< endpoint dram reserved memory in bytes for firmware */
  uint64_t ep_total_model_occupancy_size;    /**< endpoint dram memory occupied by all the active model artefacts in bytes */
  uint64_t ep_total_tensor_occupancy_size;   /**< endpoint dram memory occupied by all the active model tensors in bytes */
} dv_endpoint_dram_statistics_t;

/**
 * @brief Represents inference queue depth and latency statistics for a single endpoint.
 *
 * Use @c wait_time to estimate whether a new inference request will be picked up
 * immediately or queued behind existing requests. If @c occupancy_count approaches
 * @c length, the queue is near capacity and submitting additional requests may block.
 */
typedef struct dv_inference_queue_statistics {
  int occupancy_count; /**< Number of inference queue slots occupied with inference request for the endpoint */
  int length;          /**< length of the inference queue for the endpoint */
  float wait_time;     /**< waiting time in mili secs for the new inference request to get picked up by endpoint */
} dv_inference_queue_statistics_t;

/**
 * @brief Represents resource utilization statistics for a single model active on an endpoint.
 *
 * Use @c model_handle to correlate these statistics with a specific @ref dv_model_t
 * object returned by a model load API — compare against the @c handle member of
 * @ref dv_model_t. Statistics are reported for all models loaded across all clients,
 * not just the calling client.
 */
// Model statistics per endpoint
typedef struct dv_model_statistics {
  uint32_t model;                                    /**< model handle */
  uint32_t active_input_tensors_count;               /**< number for active model input tensor(s) present in an endpoint */
  uint32_t active_output_tensors_count;              /**< number for active model output tensor(s) present in an endpoint */
  uint32_t active_inferences_count;                  /**< number for active model inference request queued in an endpoint */
  uint32_t model_total_dram_occupancy_size;          /**< total endpoint dram occupancy in bytes by model artefacts */
  uint32_t model_total_input_tensor_occupancy_size;  /**< total endpoint dram occupancy in bytes by model input tensors */
  uint32_t model_total_output_tensor_occupancy_size; /**< total endpoint dram occupancy in bytes by model output tensors */
  dv_model_handle_t *model_handle;                   /**< void* handle which can be compared to `handle` member of dv_model_t */
} dv_model_statistics_t;

/**
 * @brief Represents a runtime statistics snapshot for a single endpoint.
 *
 * Use this to monitor endpoint health, thermal state, and clock configuration
 * during production workloads. Check @c state before submitting inference — if the
 * endpoint is in @c DV_ENDPOINT_STATE_FAULT or @c DV_ENDPOINT_STATE_THERMAL_ACTIVE_SLOW,
 * throughput may be degraded or inference may fail. Use @c ep_temp and @c ep_core_voltage
 * to detect thermal or power anomalies in long-running deployments.
 */
// Endpoint statistics
typedef struct dv_endpoint_statistics {
  dv_endpoint_t *ep;                              /**< endpoint handle */
  dv_endpoint_state_t state;                      /**< endpoint state */
  int ep_sys_clk;                                 /**< endpoint system core clock in MHz */
  int ep_dram_clk;                                /**< endpoint dram clock in MHz */
  float ep_core_voltage;                          /**< average of endpoint core voltage across all measurement point in hardware in volts */
  float ep_temp;                                  /**< average of endpoint temperature across all measurement point in hardware in degree celsius */
  int num_inference_queues;                       /**< number of inference queues available for the endpoint [unsupported] */
  dv_inference_queue_statistics_t *ep_infq_stats; /**< inference queue statistics for the endpoint [unsupported] */
  int num_active_models;                          /**< number of active models present in endpoint [unsupported] */
  dv_model_statistics_t *model_stats;             /**< statistics for all models active on the endpoint [unsupported] */
  dv_endpoint_dram_statistics_t ep_dram_stats;    /**< endpoint dram statistics */
  dv_endpoint_power_state_t ep_power_state;       /**< endpoint power state [unsupported] */
  uint32_t ep_soft_reset_count;                   /**< endpoint soft reset count, non zero for usb devices */
  int ep_sbp_clk;                                 /**< endpoint sbp clock in MHz */
  int ep_nnp_clk;                                 /**< endpoint nnp clock in MHz */
} dv_endpoint_statistics_t;

// DV model input pre-processing parameters
/**
 * Usage:
 * for quantization and dequantization of inputs/outputs is different for different qmodes
 * for qmode - 0, 1, 2 and 3:
 *    quantized_value = round(float_value * qn)
 *    dequantized_value = quantized_value / qn
 * for qmode - 9:
 *    quantized_value = round(float_value / qn) + offset
 *    dequantized_value = (float_value - offset )* qn
 */
typedef struct dv_model_input_preprocess_param {
  float qn;                  /**< quantization scale*/
  float *scale;              /**< per channel scale for nch<=3, else null */
  float *mean;               /**< per channel mean for nch<=3, else null */
  bool aspect_resize;        /**< aspect ratio based resize */
  bool mirror;               /**< mirror effect */
  bool center_crop;          /**< center crop */
  bool bgr_to_rgb;           /**< convert BGR to RGB */
  int interpolation;         /**< interpolation method supported by OpenCV */
  bool is_signed;            /**< if true, input data is in signed range [-128, 128]; if false, input data is in unsigned range [0, 255] */
  int bpp;                   /**< bytes per pixel */
  float output_scale;        /**< output scale */
  float aspect_resize_scale; /**< aspect resize scaling factor */
  int offset;                /**< offset for asymmetric quantization */
  int qmode;                 /**< quantization mode*/
} dv_model_input_preprocess_param_t;

/// Model Input Tensor Parameters
typedef struct dv_model_input_param {
  dv_model_input_preprocess_param_t *preprocess_param;
  int layer_id;               /**< input layer id */
  int blob_id;                /**< input blob id within the layer */
  char *layer_name;           /**< input layer name */
  char *blob_name;            /**< input blob name within the layer */
  char *layer_type;           /**< input layer type */
  char *layout;               /**< input layout */
  int size;                   /**< tensor size in bytes */
  int width;                  /**< tensor width */
  int height;                 /**< tensor height */
  int depth;                  /**< depth dimension */
  int nch;                    /**< number of channels */
  int bpp;                    /**< bytes per pixel */
  int batch_size;             /**< batch size */
  int num;                    /**< num */
  char *src_graph_layer_name; /**<source graph input layer name */
} dv_model_input_param_t;

/// Model output post processing parameters
typedef struct dv_model_output_postprocess_param {
  float qn;              /**< output quantization parameter */
  bool is_struct_format; /**< output is structured or not */
  bool is_float;         /**< output is float type */
  bool is_signed;        /**< output is signed or not */
  float output_scale;    /**< output scale for asymmetric quantization */
  int offset;            /**< offset for asymmetric quantization */
} dv_model_output_postprocess_param_t;

/// Model output parameters
typedef struct dv_model_output_param {
  dv_model_output_postprocess_param_t *postprocess_param; /**< */
  int layer_id;                                           /**< layer id */
  int blob_id;                                            /**< input blob id within the layer */
  int fused_parent_id;                                    /**< layer fused parent id */
  char *layer_name;                                       /**< layer name */
  char *blob_name;                                        /**< input blob name within the layer */
  char *layer_fused_parent_name;                          /**< layer fused parent name */
  char *layer_type;                                       /**< layer type */
  char *layout;                                           /**< output layout */
  int size;                                               /**< layer size in bytes */
  int width;                                              /**< layer width in pixels */
  int height;                                             /**< layer height in pixels */
  int depth;                                              /**< layer depth in pixels */
  int nch;                                                /**< number of channels */
  int bpp;                                                /**< bytes per pixel */
  int num_classes;                                        /**< number of classes for which model is trained on */
  dv_layer_output_type_t layer_output_type;               /**< output type of layer */
  int num;                                                /**< num dimension >*/
  int max_dynamic_id;                                     /**< max batch id */
  char *src_graph_layer_name;                             /**< source graph output layer name */
  int has_nms_parent;                                     /**< 1, if any parent layer is NMS*/
} dv_model_output_param_t;

/**
 * @brief Represents power and performance estimates reported by the Network Compiler.
 *
 * Use these values to compare compiler-predicted throughput against observed runtime
 * performance. A large gap between @c ips and measured inference rate may indicate
 * thermal throttling, memory pressure, or suboptimal batch configuration.
 */
// Power and performance reported by DVNC(Kinara Network Compiler)
typedef struct dv_compiler_statistics {
  char *config_name;   /**< DV1 config name, governed on ep system core clock */
  float cycles;        /**< total cycles estimated by compiler */
  float ips;           /**< inference per seconds estimated by compiler */
  float ddr_bandwidth; /**< ep dram estimated by compiler */
} dv_compiler_statistics_t;

/**
 * @brief Represents the hardware architecture and model category for a loaded model.
 *
 * Set this correctly in @ref dv_model_load_options_t when using
 * @ref dv_model_load_from_file_with_options or @ref dv_model_load_from_blob_with_options.
 * An incorrect model type will cause the proxy to use the wrong inference pipeline,
 * resulting in silent output corruption or inference failures.
 *
 * @var DV_MODEL_TYPE::DV_MODEL_TYPE_ARA1_CNN
 *   CNN model targeting the ARA1 hardware architecture.
 * @var DV_MODEL_TYPE::DV_MODEL_TYPE_ARA2_CNN
 *   CNN model targeting the ARA2 hardware architecture.
 * @var DV_MODEL_TYPE::DV_MODEL_TYPE_ARA2_LLM
 *   LLM model on ARA2 using dynamic quantization v1 (e.g., Qwen models).
 * @var DV_MODEL_TYPE::DV_MODEL_TYPE_ARA2_LLM_DYN_V2
 *   LLM model on ARA2 using the latest dynamic quantization scheme (excludes dynamic quant v1 models).
 */
typedef enum DV_MODEL_TYPE {
  DV_MODEL_TYPE_ARA1_CNN = 0,        /**< CNN models to run on ARA1 chip */
  DV_MODEL_TYPE_ARA2_CNN = 1,        /**< CNN models to run on ARA2 chip */
  DV_MODEL_TYPE_ARA2_LLM = 2,        /**< this is for dynamic quant v1 qwen models */
  DV_MODEL_TYPE_ARA2_LLM_DYN_V2 = 3, /**< latest llm models [except for dynmaic quant v1 models] */
} dv_model_type_t;

/**
 * @brief Represents options controlling model load behavior; passed to dv_model_load_*_with_options APIs.
 *
 * Use this struct instead of the legacy @ref dv_model_load_from_file or @ref dv_model_load_from_blob
 * APIs when you need fine-grained control over caching, async loading, or model type selection.
 * Always set @c model_type explicitly — the default of @c DV_MODEL_TYPE_ARA2_CNN is incorrect
 * for LLM models and will result in inference failures.
 */
// arguments to be passed to dv_model_load_*_with_options APIs
typedef struct dv_model_load_options {
  char *model_name;                   /**< model name */
  dv_model_priority_level_t priority; /**< priority of the model [unused] */
  bool cache;                         /**< if true, the model is cached on disk */
  bool async;                         /**< if true, the model load API immediately return \see
                                         dv_model_load_wait_for_completion */
  dv_model_type_t model_type;         /**< specify the model type, if not specified it will be  DV_MODEL_TYPE_ARA2_CNN*/
  bool send_model_filepath_to_proxy;  /**< if true, send filepaht from client lib to proxy, imporves host memory consumpion
                                         use this only if proxy and client lib have direct access to file(both are on same machine, or share filesystem)> */
} dv_model_load_options_t;

/**
 * @brief Represents the inference execution mode for a submitted inference request.
 *
 * For CNN inference, use @c DV_INFER_TYPE_ARA2_CNN (default). For LLM workflows,
 * select the appropriate phase — prompt processing must precede token generation.
 * Using @c DV_INFER_TYPE_LLM_TOKEN_GENERATION without a prior prompt processing
 * inference on the same model results in undefined output.
 *
 * @var DV_INFER_TYPE::DV_INFER_TYPE_ARA1_CNN
 *   Standard CNN inference on ARA1 hardware.
 * @var DV_INFER_TYPE::DV_INFER_TYPE_ARA2_CNN
 *   Standard CNN inference on ARA2 hardware.
 * @var DV_INFER_TYPE::DV_INFER_TYPE_LLM_PROMPT_PROCESSING
 *   LLM initial prompt processing (prefill) phase.
 * @var DV_INFER_TYPE::DV_INFER_TYPE_LLM_FOLLOWUP_PROMPT_PROCESSING
 *   LLM follow-up prompt processing for multi-turn or continued inference.
 * @var DV_INFER_TYPE::DV_INFER_TYPE_LLM_TOKEN_GENERATION
 *   LLM autoregressive token generation (decode) phase.
 */
typedef enum DV_INFER_TYPE {
  DV_INFER_TYPE_ARA1_CNN = 0,
  DV_INFER_TYPE_ARA2_CNN = 1,
  DV_INFER_TYPE_LLM_PROMPT_PROCESSING = 2,
  DV_INFER_TYPE_LLM_FOLLOWUP_PROMPT_PROCESSING = 3,
  DV_INFER_TYPE_LLM_TOKEN_GENERATION = 4,

} dv_infer_type_t;

/**
 * @brief Represents options controlling the behavior of an inference request submission.
 *
 * Pass this to @ref dv_infer_sync_with_options or @ref dv_infer_async_with_options
 * when you need LLM-specific inference control or statistics collection.
 * For standard CNN inference, the default zero-initialized struct is sufficient.
 * Do not use @c active_tokens or @c valid_tokens for CNN inference — they are
 * only meaningful for LLM token generation and prompt processing phases.
 *
 * @var dv_infer_options_t::enable_stats
 *   If true, inference statistics are collected for this request.
 * @var dv_infer_options_t::infer_type
 *   Inference execution mode; defaults to DV_INFER_TYPE_ARA2_CNN if not set.
 * @var dv_infer_options_t::active_tokens
 *   Number of active tokens for LLM inference requests.
 * @var dv_infer_options_t::valid_tokens
 *   Number of valid tokens in the input for LLM inference requests.
 * @var dv_infer_options_t::tokens_to_skip
 *   Number of image or video tokens in the prompt to skip.
 */
typedef struct dv_infer_options {
  // bool async;
  bool enable_stats;
  // uint64_t timeout_ms;
  dv_infer_type_t infer_type; /**< specify the model type, if not specified it will be  DV_MODEL_TYPE_ARA2_CNN*/
  uint64_t active_tokens;
  uint32_t valid_tokens;
  uint32_t tokens_to_skip;  // number of image, video tokens in the prompt.
} dv_infer_options_t;

/**
 * @brief Represents parameters passed to @ref dv_endpoint_get_statistics_with_options to control
 *        the type and filtering of returned endpoint statistics.
 *
 * Use @c state_type to request a specific category of state information from the server.
 * Use @c temp_threshold to filter endpoints by temperature — only endpoints at or above
 * this threshold (in degrees Celsius) are included in the response, which is useful when
 * monitoring for thermal events and only interested in endpoints running hot.
 *
 * @var dv_stats_options_t::state_type
 *   Identifies the category of state information requested.
 * @var dv_stats_options_t::temp_threshold
 *   Temperature threshold in degrees Celsius used to filter the statistics response.
 */
// sending parameters for stats api
typedef struct dv_stats_options {
  uint32_t state_type;
  uint32_t temp_threshold;
} dv_stats_options_t;

/**
 * @brief Represents LLM-specific model parameters describing token configuration and memory layout.
 *
 * This struct is populated automatically by the client library after a successful LLM model load
 * and is accessible via @ref dv_model_t::llm_params. Do not modify these fields manually —
 * they reflect the model's compiled configuration and are used internally by the inference pipeline.
 * Use @c max_num_tokens to validate that your prompt length does not exceed the model's capacity
 * before submitting inference requests.
 *
 * @var dv_model_llm_params_t::vocab_size
 *   Vocabulary size of the model.
 * @var dv_model_llm_params_t::embedding_size
 *   Hidden (embedding) dimension size.
 * @var dv_model_llm_params_t::input_precision
 *   Data precision of model inputs; default is 8.
 * @var dv_model_llm_params_t::output_precision
 *   Data precision of model outputs; default is 32.
 * @var dv_model_llm_params_t::max_num_tokens
 *   Maximum number of tokens supported by the model.
 * @var dv_model_llm_params_t::is_dynamic
 *   If non-zero, the model is a dynamic LLM; default is true.
 * @var dv_model_llm_params_t::num_inputs
 *   Number of inputs for the model; default is 1.
 * @var dv_model_llm_params_t::pad_token_id
 *   Token ID used for padding.
 * @var dv_model_llm_params_t::eos_token_id
 *   Token ID marking end of sequence.
 * @var dv_model_llm_params_t::bos_token_id
 *   Token ID marking beginning of sequence.
 * @var dv_model_llm_params_t::embedding_lookup_addr
 *   Device address for input embeddings.
 * @var dv_model_llm_params_t::embedding_lookup_scale_addr
 *   Device address for input embedding scales.
 * @var dv_model_llm_params_t::is_speculative
 *   If non-zero, the model supports speculative decoding.
 * @var dv_model_llm_params_t::max_prompt_input_size
 *   Maximum size in bytes of a prompt input.
 * @var dv_model_llm_params_t::max_token_input_size
 *   Maximum size in bytes of a token input.
 * @var dv_model_llm_params_t::max_output_size
 *   Maximum size in bytes of the model output.
 * @var dv_model_llm_params_t::is_host_specd
 *   If non-zero, the model supports host-driven speculative decoding.
 */
typedef struct dv_model_llm_params {
  uint32_t vocab_size;                   // Vocab size of the model
  uint32_t embedding_size;               // Hidden size
  uint32_t input_precision;              // Data precision of inputs default 8
  uint32_t output_precision;             // Data precision of outputs default 32
  uint32_t max_num_tokens;               // Max supported tokens count.
  uint32_t is_dynamic;                   // Specifies if model is dynamic llm default true
  uint32_t num_inputs;                   // Number of inputs for the model, default is 1.
  uint32_t pad_token_id;                 // Model padding token id
  uint32_t eos_token_id;                 // Model end of sequence token id
  uint32_t bos_token_id;                 // Model begin of sequence token id
  uint64_t embedding_lookup_addr;        // address for input embeddings
  uint64_t embedding_lookup_scale_addr;  // address for input embedding scales.
  uint32_t is_speculative;               // Specifies if model  is specd.
  uint64_t max_prompt_input_size;        // max size of prompt input.
  uint64_t max_token_input_size;         // Max size of input.
  uint64_t max_output_size;              // Max size of output.
  uint8_t is_host_specd;                 // Specifies if model supports speculations from host.
} dv_model_llm_params_t;

/**
 * @brief Represents configuration parameters for updating LLM sampling and speculative decoding settings.
 *
 * Pass this to @ref dv_model_set_llm_cfg_params to tune sampling behavior at runtime
 * without reloading the model. MCP counts (@c target_token_post_mcp etc.) control
 * speculative decoding batch sizes and should only be set when the model was compiled
 * with speculative decoding support.
 *
 * @var dv_llm_cfg_upd_req_t::top_k
 *   Top-K value for sampling.
 * @var dv_llm_cfg_upd_req_t::top_p
 *   Top-P (nucleus sampling) probability threshold.
 * @var dv_llm_cfg_upd_req_t::temperature
 *   Sampling temperature controlling output randomness.
 * @var dv_llm_cfg_upd_req_t::repetition_penalty
 *   Penalty factor applied to discourage repeated tokens.
 * @var dv_llm_cfg_upd_req_t::target_token_post_mcp
 *   Target token count post-MCP for the target model in token generation phase.
 * @var dv_llm_cfg_upd_req_t::target_token_pre_mcp
 *   Target token count pre-MCP for the target model in token generation phase.
 * @var dv_llm_cfg_upd_req_t::target_prompt_post_mcp
 *   Target prompt count post-MCP for the target model in prompt processing phase.
 * @var dv_llm_cfg_upd_req_t::target_prompt_pre_mcp
 *   Target prompt count pre-MCP for the target model in prompt processing phase.
 * @var dv_llm_cfg_upd_req_t::draft_token_post_mcp
 *   Draft token count post-MCP for the draft model in speculative decoding.
 * @var dv_llm_cfg_upd_req_t::draft_token_pre_mcp
 *   Draft token count pre-MCP for the draft model in speculative decoding.
 * @var dv_llm_cfg_upd_req_t::draft_prompt_post_mcp
 *   Draft prompt count post-MCP for the draft model in speculative decoding.
 * @var dv_llm_cfg_upd_req_t::draft_prompt_pre_mcp
 *   Draft prompt count pre-MCP for the draft model in speculative decoding.
 */
typedef struct {
  uint32_t top_k;                  /**< sampling llm parameter top_k */
  float top_p;                     /**< nucleus sampling llm parameter top_p */
  float temperature;               /**< llm parameter temperature */
  float repetition_penalty;        /**< llm parameter repetition_penalty */
  uint32_t target_token_post_mcp;  /**< token post-processing targeted in mcp or host */
  uint32_t target_token_pre_mcp;   /**< token pre-processing targeted in mcp or host */
  uint32_t target_prompt_post_mcp; /**< prompt post-processing targeted in mcp or host */
  uint32_t target_prompt_pre_mcp;  /**< prompt pre-processing targeted in mcp or host */
  uint32_t draft_token_post_mcp;   /**< TRUE(1) for specd model */
  uint32_t draft_token_pre_mcp;    /**< TRUE(1) for specd model */
  uint32_t draft_prompt_post_mcp;  /**< TRUE(1) for specd model */
  uint32_t draft_prompt_pre_mcp;   /**< TRUE(1) for specd model */
} dv_llm_cfg_upd_req_t;

/**
 * @brief Represents a loaded model and its associated metadata, session, and endpoint bindings.
 *
 * This object is returned by all model load APIs and must be kept alive for the duration
 * of any inference requests that use it. Do not free or modify this struct directly —
 * use @ref dv_model_unload to release it. For LLM models, check @c llm_params for
 * token capacity and precision information before submitting inference requests.
 */
// Model object
typedef struct dv_model {
  dv_model_handle_t *handle;                   /**< model handle, managed by client library */
  dv_session_t *session;                       /**< session handle on which model is loaded */
  dv_endpoint_t *endpoint;                     /**< endpoint handle on which model is loaded */
  dv_version_t version;                        /**< compiled model version */
  char *name;                                  /**< model name provided by user */
  dv_model_type_t model_type;                  /**< model type */
  char *internal_name;                         /**< model name embedded during compilation */
  int num_inputs;                              /**< number of inputs needed by model */
  int num_outputs;                             /**< number of outputs produced by model */
  dv_model_priority_level_t priority;          /**< model priority as set by user [unused]*/
  dv_model_input_param_t *input_param;         /**< list of model specific input params (usefull for pre-processing) */
  dv_model_output_param_t *output_param;       /**< list of model specific output params (usefull for post-processing) */
  dv_model_llm_params_t *llm_params;           /**< list of llm params, valid if model is llm model*/
  int num_compiler_config;                     /**< [unsupported] */
  dv_compiler_statistics_t *compiler_stats;    /**< [unsupported] */
  dv_model_load_options_t *model_load_options; /**< [unsupported] */
  bool cp_layer;                               /**< [unsupported] */
} dv_model_t;

/**
 * @brief Represents detailed timing and hardware counter statistics for a completed inference request.
 *
 * Populate this by passing @c enable_stats = true in the inference request options.
 * Use @c inference_execution_time and @c ep_hw_total_inference_cycles to profile
 * model performance on hardware. Use @c input_transfer_time and @c output_transfer_time
 * to identify data transfer bottlenecks between host and endpoint DRAM.
 * Note: @c ep_queue_submission_time is currently unsupported and will always be -1.
 */
// Inference statistics
typedef struct dv_infer_statistics {
  int ep_hw_sys_clk;                                /**< endpoint hardware system core clock in MHz */
  int ep_hw_nnp_clk;                                /**< endpoint hardware external nnp clock in MHz */
  int ep_hw_sbp_clk;                                /**< endpoint hardware external sbp clock in MHz */
  int ep_hw_dram_clk;                               /**< endpoint hardware external dram clock in MHz */
  uint32_t ep_hw_total_inference_cycles;            /**< total cycles taken to compute
                                                       inference in hardware, including
                                                       floating point computation */
  uint32_t ep_hw_fp_cycles;                         /**< cycles taken to compute floating point
                                                       operation in hardware */
  float input_transfer_time;                        /**< time taken in microseconds to transfer
                                                       input(s) from host dram to ep hardware dram */
  float output_transfer_time;                       /**< time taken in microseconds to transfer
                                                       output(s) from ep hardware dram to host dram */
  float ep_queue_submission_time;                   /**< time taken in microseconds to submit
                                                       inference request to ep hardware
                                                       As of now this is not supported and assigned a default value -1*/
  uint32_t cumulative_replay_count;                 /**< Total number of infer retries occured  per session of proxy */
  uint32_t current_replay_count;                    /**< Total number of infer retries occured  per inference */
  struct timespec input_transfer_start_time_stamp;  /**< time stamp when input
                                                        transfer started*/
  struct timespec output_transfer_start_time_stamp; /**< time stamp when output
                                                       transfer started*/
  struct timespec inference_start_time_stamp;       /**< time stamp when inference
                                                      went into NNP queue */
  float inference_execution_time;                   /**< time taken for inference execution in
                                                       microseconds*/
  uint32_t input_ddr_address;                       /**< input ddr address **/
  uint32_t output_ddr_address;                      /**< output ddr address **/
} dv_infer_statistics_t;

/**
 * @brief Represents LLM-specific information returned as part of a completed inference response.
 *
 * Valid only for LLM inference requests. Use @c llm_infer_resp_num_valid_tokens to determine
 * how many output tokens were actually generated — the output blob may be larger than the
 * valid token count if the model pre-allocates maximum output capacity.
 *
 * @var dv_infer_llm_info_t::llm_infer_resp_num_valid_tokens
 *   Number of valid tokens in the LLM inference response.
 */
typedef struct dv_infer_llm_info {
  uint32_t llm_infer_resp_num_valid_tokens;
} dv_infer_llm_info_t;

/**
 * @brief Represents a single inference request and its associated runtime state.
 *
 * This object is returned by all inference submission APIs and must be kept alive
 * until the inference completes. Check @c status to determine the current state.
 * After completion, read results from @c op_blob_list and optionally inspect
 * @c stats if statistics were enabled. Always free this object with @ref dv_infer_free
 * after use — failure to do so will leak memory in the client library.
 * For LLM inference, check @c llm_infer_info for valid token count before
 * reading the output blob.
 */
// Inference request object
typedef struct dv_infer_request {
  dv_infer_request_handle_t *handle; /**< private handle, managed by client library */
  dv_session_t *session;             /**< session for which inference is submitted */
  dv_endpoint_t *ep_queued;          /**< endpoint for which inference is queued. */
  dv_endpoint_t *ep_submitted;       /**< when inference request is queued on group of
                                        endpoints, this provide endpoint info on which
                                        inference is submitted. */
  dv_model_t *model;                 /**< model handle for inference request */
  dv_blob_t *ip_blob_list;           /**< input blob list */
  dv_blob_t *op_blob_list;           /**< output blob list */
  dv_inference_status_t status;      /**< inference run status */
  dv_infer_statistics_t *stats;      /**< inference stats */
  dv_infer_llm_info_t *llm_infer_info; /**< LLM-specific inference response info; valid only for LLM inference requests */
} dv_infer_request_t;

/********************************** DV Client APIs
 * *************************************************/

/**
 * @brief Converts a status code to its human-readable string representation.
 *
 * Use this when logging errors or displaying status information in diagnostic output.
 * The returned string is statically allocated and must not be freed or modified by
 * the caller. If an unrecognized or out-of-range status code is passed, the function
 * returns the string @c "DV_STATUS_CODE_UNKNOWN_L" rather than crashing or returning NULL.
 *
 * @param[in]  status_code  Status code to stringify.
 * @return                  Null-terminated string name of the status code,
 *                          or @c "DV_STATUS_CODE_UNKNOWN_L" if unrecognized.
 */
EXPORT
const char *dv_stringify_status_code(dv_status_code_t status_code);

/**
 * @brief Sets the log verbosity level for the DV client library.
 *
 * Use this at application startup to control how much diagnostic output the
 * client library emits. In production, prefer @c DV_CLIENT_LOG_LEVEL_WARN or
 * @c DV_CLIENT_LOG_LEVEL_ERROR to reduce log noise. Use @c DV_CLIENT_LOG_LEVEL_DEBUG
 * or @c DV_CLIENT_LOG_LEVEL_TRACE during development or when diagnosing failures.
 * Set to @c DV_CLIENT_LOG_LEVEL_OFF to silence all client library logs.
 * Only messages at or above the specified level are output.
 *
 * @param[in]  log_level  Desired log level as defined by @ref DV_CLIENT_LOG_LEVEL.
 * @return                DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_client_set_log_level(dv_client_log_level_t log_level);

/********************************** DV Session APIs
 * *************************************************/

/**
 * @brief Creates a session to the inference proxy server over a Unix domain socket.
 *
 * Use this when the proxy and client are running on the same machine — Unix domain
 * sockets have lower latency and overhead than TCP for local communication.
 * Prefer this over @ref dv_session_create_via_tcp_ipv4_socket for same-host deployments.
 * Note that SHM registration via @ref dv_shmfd_register is only supported over Unix
 * domain sockets, not TCP.
 *
 * @param[in]  socket_file_path  Path to the Unix domain socket file.
 * @param[out] session           Session handle returned on success.
 * @return                       DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_session_create_via_unix_socket(const char *socket_file_path, dv_session_t **session);

/**
 * @brief Creates a session to the inference proxy server using a Windows named pipe.
 *
 * Use this only on Windows hosts where Unix domain sockets are unavailable.
 * On Linux, use @ref dv_session_create_via_unix_socket instead.
 *
 * @param[in]  named_pipe  Name of the Windows named pipe.
 * @param[out] session     Session handle returned on success.
 * @return                 DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_session_create_via_named_pipe(const char *named_pipe, dv_session_t **session);

/**
 * @brief Creates a session to the inference proxy server over a TCP/IPv4 socket.
 *
 * Use this when the proxy is running on a different machine from the client,
 * or in containerized deployments where Unix domain socket sharing is not available.
 * For same-host deployments, prefer @ref dv_session_create_via_unix_socket for
 * lower latency. Note that SHM registration via @ref dv_shmfd_register is not
 * supported over TCP — use raw pointer blobs instead for TCP sessions.
 *
 * @param[in]  tcp_ip_addr  IPv4 address of the inference proxy server.
 * @param[in]  port         TCP port number on which the server is listening.
 * @param[out] session      Session handle returned on success.
 * @return                  DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_session_create_via_tcp_ipv4_socket(const char *tcp_ip_addr, int port, dv_session_t **session);

/**
 * @brief Closes an open session to the inference proxy server.
 *
 * Call this when the application is done using the proxy — for example, at shutdown.
 * Ensure all in-flight inference requests have completed before closing the session;
 * closing a session with pending inferences will cause those requests to fail.
 * The session handle must not be used after this call.
 *
 * @param[in]  session  Session handle to close.
 * @return              DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_session_close(dv_session_t *session);

/********************************** DV SharedFD API
 * ***************************************/

/**
 * @brief Registers a file descriptor with the inference proxy server for shared memory access.
 *
 * Use this to avoid redundant host-to-device data copies across multiple inference
 * requests that use the same input buffer. Once registered, use the returned descriptor
 * as a @ref DV_BLOB_TYPE_SHM_DESCRIPTOR blob in inference calls.
 * Do not use this over TCP sessions — SHM registration is only supported over Unix
 * domain sockets.
 *
 * The server maps the fd into a server-managed shared buffer identified by an opaque
 * buf_id stored in @c shm_desc->handle. This buf_id is server-global: if client A
 * registers a descriptor and client B submits an inference using the same handle value,
 * both clients will access the same underlying server-side buffer with no isolation
 * between them. The registering client is responsible for coordinating concurrent
 * access across clients to avoid data corruption.
 *
 * The descriptor must be unregistered via @ref dv_shmfd_unregister when no longer needed.
 *
 * @param[in]  session   Session handle.
 * @param[in]  fd        File descriptor to register.
 * @param[in]  size      Size in bytes to map.
 * @param[in]  offset    Offset within the file to begin mapping.
 * @param[in]  fd_type   Type hint for the file descriptor provided to the server (reserved).
 * @param[out] shm_desc  Shared memory descriptor returned on success.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_shmfd_register(dv_session_t *session, int fd, uint32_t size, uint32_t offset, int fd_type, dv_shm_descriptor_t **shm_desc);

/**
 * @brief Unregisters a previously registered shared memory file descriptor from the server.
 *
 * Call this when the shared buffer is no longer needed for inference. Do not unregister
 * a descriptor while another client may still be using its buf_id in an active inference
 * request, as the server-side mapping will be released immediately.
 * The descriptor must not be used after this call.
 *
 * @param[in]  shm_desc  Shared memory descriptor to unregister.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_shmfd_unregister(dv_shm_descriptor_t *shm_desc);

/********************************** DV Endpoint APIs
 * *************************************************/

/**
 * @brief Retrieves the list of all endpoints currently connected to the inference proxy server.
 *
 * Use this to discover available hardware before selecting an endpoint for model loading
 * or inference. Call this once after session creation and cache the result — the list
 * does not change unless endpoints are added or removed from the proxy.
 *
 * Memory for the endpoint list is allocated statically by the client library and must
 * NOT be freed by the caller.
 *
 * @param[in]  session   Session handle.
 * @param[out] ep_list   Array of endpoint handles returned by the proxy.
 * @param[out] ep_count  Number of endpoints in the returned list.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_get_list(dv_session_t *session, dv_endpoint_t **ep_list, int *ep_count);

/**
 * @brief Retrieves a predefined default endpoint group from the inference proxy server.
 *
 * Use this when you want to submit inference to all endpoints of a given interface type
 * without manually constructing a group. Prefer this over @ref dv_endpoint_create_group
 * when you do not need fine-grained endpoint selection. Use @c DV_ENDPOINT_DEFAULT_GROUP_ALL
 * for maximum throughput across all connected devices, or @c DV_ENDPOINT_DEFAULT_GROUP_PCIE
 * / @c DV_ENDPOINT_DEFAULT_GROUP_USB to restrict to a specific interface type.
 * Memory for the group configuration is managed by client library, and should NOT be deallocated.
 *
 * @param[in]  session  Session handle.
 * @param[in]  grp      Default group type as defined by @ref DV_ENDPOINT_DEFAULT_GROUP.
 * @param[out] ep_grp   Endpoint group handle returned on success.
 * @return              DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_get_default_group(dv_session_t *session, dv_endpoint_default_group_t grp, dv_endpoint_t **ep_grp);

/**
 * @brief Creates a custom endpoint group from a caller-supplied list of endpoints.
 *
 * Use this when you need to load a model or submit inference on a specific subset of
 * endpoints — for example, isolating certain devices for a particular workload.
 * Do not use this if a predefined group via @ref dv_endpoint_get_default_group satisfies
 * your requirements. Memory for the group configuration is allocated by the API and
 * must be freed using @ref dv_endpoint_free_group.
 *
 * @param[in]  session   Session handle.
 * @param[in]  ep_list   Array of endpoint handles to include in the group.
 * @param[in]  ep_count  Number of endpoints in @p ep_list.
 * @param[out] ep_grp    Endpoint group handle returned on success.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_create_group(dv_session_t *session, dv_endpoint_t **ep_list, int ep_count, dv_endpoint_t **ep_grp);

/**
 * @brief Frees a custom endpoint group previously obtained from the proxy.
 *
 * Call this when the endpoint group is no longer needed. Do not free a group while a
 * model is still loaded on it or while inference is in flight on that group.
 * The handle must not be used after this call.
 *
 * @param[in]  ep_grp  Endpoint group handle to free.
 * @return             DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_free_group(dv_endpoint_t *ep_grp);

/**
 * @brief Queries the current operational state of an endpoint.
 *
 * Use this before loading a model or submitting inference to confirm the endpoint
 * is in a healthy state. If the endpoint is in @c DV_ENDPOINT_STATE_FAULT, model
 * loading will fail and inference will not be accepted. If the endpoint is in
 * @c DV_ENDPOINT_STATE_THERMAL_ACTIVE_SLOW or @c DV_ENDPOINT_STATE_ACTIVE_SLOW,
 * inference will still run but at reduced throughput.
 *
 * @param[in]  session  Session handle.
 * @param[in]  ep       Endpoint handle to query.
 * @param[out] state    Current state of the endpoint as defined by @ref DV_ENDPOINT_STATE.
 * @return              DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_check_status(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_state_t *state);

/********************************** DV Endpoint stats APIs
 * *************************************************/

/**
 * @brief Retrieves a snapshot of DRAM usage statistics for the specified endpoint or endpoint group.
 *
 * Use this to check memory usage information for an endpoint.
 * Passing NULL for @p ep returns DRAM statistics for all endpoints connected to the server.
 * Memory for the returned statistics is allocated by the API, and must be freed
 * using @ref dv_endpoint_free_dram_statistics.
 *
 * @param[in]  session         Session handle.
 * @param[in]  ep              Endpoint or endpoint group handle, or NULL for all endpoints.
 * @param[out] ep_dram_stats   Array of DRAM statistics structures returned by the server.
 * @param[out] ep_count        Number of endpoints for which statistics are returned.
 * @return                     DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_get_dram_statistics(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_dram_statistics_t **ep_dram_stats, int *ep_count);

/**
 * @brief Frees memory allocated by @ref dv_endpoint_get_dram_statistics.
 *
 * Always call this after processing the DRAM statistics to avoid memory leaks.
 * Do not access @p ep_dram_stats after this call.
 *
 * @param[in]  ep_dram_stats  Pointer to the DRAM statistics array to free.
 * @param[in]  count          Number of elements in the array.
 * @return                    DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_free_dram_statistics(dv_endpoint_dram_statistics_t *ep_dram_stats, int count);

/**
 * @brief Retrieves a statistics snapshot for the specified endpoint or endpoint group,
 *        with additional inference options to control the statistics retrieval behavior.
 *
 * Use this variant over @ref dv_endpoint_get_statistics when you need to pass
 * inference options — for example, to filter statistics by inference type for LLM workloads.
 * For standard CNN workloads without special filtering, use @ref dv_endpoint_get_statistics.
 * Passing NULL for @p ep returns statistics for all endpoints connected to the server.
 * Memory for the statistics is allocated by the API and must be freed using
 * @ref dv_endpoint_free_statistics.
 *
 * The @ref dv_model_statistics_t member of @ref dv_endpoint_statistics_t exposes two
 * model identity fields: @c uint32_t model (server-generated ID, always valid) and
 * @c void* model_handle. The @c model_handle value can be compared against the @c handle
 * member of @ref dv_model_t to correlate statistics with a specific loaded model object.
 * Statistics are returned for all models loaded across all clients.
 *
 * @param[in]  session        Session handle.
 * @param[in]  ep             Endpoint or endpoint group handle, or NULL for all endpoints.
 * @param[out] ep_stats       Array of endpoint statistics structures returned by the server.
 * @param[out] ep_count       Number of endpoints for which statistics are returned.
 * @param[in]  stats_options  Option to select category of state and get info with temperature threshold to filter the statistics response.
 * @return                    DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_get_statistics_with_options(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_statistics_t **ep_stats, int *ep_count, dv_stats_options_t *stats_options);

/**
 * @brief Retrieves a statistics snapshot for the specified endpoint or endpoint group.
 *
 * Use this for general-purpose endpoint health monitoring — checking thermal state,
 * clock frequencies, DRAM usage, and active model count. Call this periodically in
 * production to detect thermal throttling (@c DV_ENDPOINT_STATE_THERMAL_ACTIVE_SLOW)
 * or fault conditions (@c DV_ENDPOINT_STATE_FAULT) before they impact inference.
 * If you need to pass inference options to filter the statistics, use
 * @ref dv_endpoint_get_statistics_with_options instead.
 * Passing NULL for @p ep returns statistics for all endpoints connected to the server.
 * Memory for the statistics is allocated by the API and must be freed using
 * @ref dv_endpoint_free_statistics.
 *
 * The @ref dv_model_statistics_t member of @ref dv_endpoint_statistics_t exposes two
 * model identity fields: @c uint32_t model (server-generated ID, always valid) and
 * @c void* model_handle. The @c model_handle value can be compared against the @c handle
 * member of @ref dv_model_t to correlate statistics with a specific loaded model object.
 * Statistics are returned for all models loaded across all clients.
 *
 * @param[in]  session   Session handle.
 * @param[in]  ep        Endpoint or endpoint group handle, or NULL for all endpoints.
 * @param[out] ep_stats  Array of endpoint statistics structures returned by the server.
 * @param[out] ep_count  Number of endpoints for which statistics are returned.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_get_statistics(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_statistics_t **ep_stats, int *ep_count);

/**
 * @brief Frees memory allocated by @ref dv_endpoint_get_statistics or
 *        @ref dv_endpoint_get_statistics_with_options.
 *
 * Always call this after processing endpoint statistics to avoid memory leaks.
 * Do not access @p ep_stats after this call.
 *
 * @param[in]  ep_stats  Pointer to the endpoint statistics array to free.
 * @param[in]  count     Number of elements in the array.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_endpoint_free_statistics(dv_endpoint_statistics_t *ep_stats, int count);

/********************************** DV Model APIs
 * *************************************************/

/**
 * @brief Loads a model from a file and transfers it to the specified endpoint.
 *
 * Use this for the simplest model load workflow when the compiled model file is
 * accessible on the local filesystem. For more control over load behavior (async
 * loading, caching, model type), use @ref dv_model_load_from_file_with_options instead.
 * If the model file is already in memory, use @ref dv_model_load_from_blob to avoid
 * an additional file read. If model loading fails on all devices represented by
 * @p endpt, the API returns an error.
 *
 * @param[in]  session          Session handle.
 * @param[in]  endpt            Endpoint or endpoint group handle to load the model onto.
 * @param[in]  model_file_path  Filesystem path to the compiled model file.
 * @param[in]  model_name       Model name .
 * @param[in]  priority         Model scheduling priority [unused].
 * @param[out] model_handle     Model object returned on success.
 * @return                      DV_SUCCESS if loaded successfully on all devices, DV_PARTIAL_SUCCESS if loaded on only some devices, DV_MODEL_LOAD_FAILURE if loading failed on all devices.
 * @note    Prefer @ref dv_model_load_from_file_s which returns a detailed @ref dv_status_t
 *          instead of a plain status code.
 */
EXPORT
dv_status_code_t dv_model_load_from_file(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);

/**
 * @brief Wrapper of dv_model_load_from_file with detailed status.
 *
 * Use this instead of @ref dv_model_load_from_file when you need a detailed status
 * object rather than a simple status code.
 *
 * @param[in]  session          Session handle.
 * @param[in]  endpt            Endpoint or endpoint group handle to load the model onto.
 * @param[in]  model_file_path  Filesystem path to the compiled model file.
 * @param[in]  model_name       Model name.
 * @param[in]  priority         Model scheduling priority [unused].
 * @param[out] model_handle     Model object returned on success.
 * @return                      DV_SUCCESS if loaded successfully on all devices, DV_PARTIAL_SUCCESS if loaded on only some devices, DV_MODEL_LOAD_FAILURE if loading failed on all devices.
 */
EXPORT
dv_status_t dv_model_load_from_file_s(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);

/**
 * @brief Loads a model from a blob and transfers it to the specified endpoint.
 *
 * Use this when the model is already resident in host memory — for example, when
 * the model was downloaded over the network or embedded in the application binary.
 * This avoids a filesystem read compared to @ref dv_model_load_from_file.
 * For more control over load behavior, use @ref dv_model_load_from_blob_with_options.
 *
 * @param[in]  session       Session handle.
 * @param[in]  endpt         Endpoint or endpoint group handle to load the model onto.
 * @param[in]  model_blob    Blob descriptor pointing to the model data in memory.
 * @param[in]  model_name    Model name .
 * @param[in]  priority      Model scheduling priority [unused].
 * @param[out] model_handle  Model object returned on success.
 * @return                   DV_SUCCESS if loaded successfully on all devices,
 *                           DV_PARTIAL_SUCCESS if loaded on only some devices,
 *                           Error Status if loading failed on all devices.
 */
EXPORT
dv_status_code_t dv_model_load_from_blob(dv_session_t *session, dv_endpoint_t *endpt, dv_blob_t *model_blob, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);

/**
 * @brief Unloads a model from its endpoint and releases associated server-side resources.
 *
 * Call this when the model is no longer needed for inference. Ensure all in-flight
 * inference requests using this model have completed before unloading; unloading a
 * model with active inference requests results in those requests failing.
 * The model handle must not be used after this call.
 *
 * @param[in]  model  Model handle to unload.
 * @return            DV_SUCCESS if unloaded successfully on all devices,
 *                    DV_PARTIAL_SUCCESS if unloaded on only some devices,
 *                    DV_MODEL_INVALID_HANDLE if the handel sent is invalid,
 *                    DV_SESSION_INVALID_HANDLE if there is no valid client (user) found,
 *                    DV_MODEL_UNLOAD_FAILURE if unloading failed on all devices.
 */
EXPORT
dv_status_code_t dv_model_unload(dv_model_t *model);

/**
 * @brief Retrieves model parameters from a file without loading the model onto an endpoint.
 *
 * Use this to inspect model metadata — input/output shapes, layer names, quantization
 * parameters — before committing to a full model load. Do not use the returned model
 * object for inference — it has no endpoint binding and will fail if passed to inference
 * APIs. Memory for the returned model object is allocated by the API and must be freed
 * using @ref dv_model_free_parameters.
 *
 * @param[in]  model_file_path  Filesystem path to the compiled model file.
 * @param[out] model            Model object populated with parameter information.
 * @return                      DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_model_get_parameters_from_file(const char *model_file_path, dv_model_t **model);

/**
 * @brief Retrieves model parameters from a blob without loading the model onto an endpoint.
 *
 * Use this to inspect model metadata when the model is already in memory, without
 * transferring it to an endpoint. Do not use the returned model object for inference —
 * it has no endpoint binding and will fail if passed to inference APIs. Memory for the
 * returned model object is allocated by the API and must be freed using
 * @ref dv_model_free_parameters.
 *
 * @param[in]  model_blob  Blob descriptor pointing to the model data in memory.
 * @param[out] model       Model object populated with parameter information.
 * @return                 DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_model_get_parameters_from_blob(dv_blob_t *model_blob, dv_model_t **model);

/**
 * @brief Frees memory allocated by @ref dv_model_get_parameters_from_file or
 *        @ref dv_model_get_parameters_from_blob.
 *
 * Always call this after inspecting model parameters to avoid memory leaks.
 * Do not call this on a model handle returned by a model load API — use
 * @ref dv_model_unload for loaded models instead.
 *
 * @param[in]  model  Model handle whose parameter memory is to be freed.
 * @return            DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_model_free_parameters(dv_model_t *model);

/**
 * @brief Sends LLM pre/post processing configuration parameters to the proxy or MCP.
 *
 * Use this to update sampling and speculative decoding settings at runtime without
 * reloading the model. Call this after model load and before inference submission.
 * Do not call this during an active inference request on the same model.
 * The @p model parameter is currently unused by the proxy but should be passed
 * for forward compatibility.
 *
 * @param[in]  session         Session handle.
 * @param[in]  ep              Endpoint handle.
 * @param[in]  model           Model handle (currently unused by proxy).
 * @param[in]  llm_cfg_update  Structure containing LLM configuration parameters to apply.
 * @return                     DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_model_set_llm_cfg_params(dv_session_t *session, dv_endpoint_t *ep, dv_model_t *model, dv_llm_cfg_upd_req_t *llm_cfg_update);
/********************************** DV Inference APIs
 * *************************************************/

/**
 * @brief Submits a synchronous inference request and blocks until completion or timeout.
 *
 * Use this for simple, sequential inference workflows where the calling thread can
 * block until the result is ready. Do not use this in latency-sensitive or
 * high-throughput pipelines — use @ref dv_infer_async instead to overlap inference
 * with host-side processing. Input and output blobs must conform to the model's
 * declared input and output parameters. Memory for the inference object is allocated
 * by the API and must be freed using @ref dv_infer_free after reading the results.
 *
 * @param[in]  session       Session handle.
 * @param[in]  endpt         Endpoint or endpoint group handle to submit inference to. Although the
 *                           model already carries the endpoint group it was loaded on, this parameter
 *                           allows the caller to narrow or override the target to a specific endpoint
 *                           or sub-group within that group for this inference request.
 * @param[in]  model         Model handle for which inference is requested.
 * @param[in]  ip_array      Array of input blob descriptors, one per model input.
 * @param[in]  op_array      Array of output blob descriptors, one per model output.
 * @param[in]  timeout       Maximum time in milliseconds to wait; pass -1 for default (60s).
 * @param[in]  enable_stats  Enable inference statistics collection (deprecated).
 * @param[out] inf_obj       Inference request object returned on completion.
 * @return                   DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_infer_sync(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, int timeout, bool enable_stats, dv_infer_request_t **inf_obj);

/**
 * @brief Submits an asynchronous inference request and returns immediately.
 *
 * Use this when you want to overlap inference execution with host-side processing.
 * This is preferred over @ref dv_infer_sync for high-throughput pipelines.
 * After submission, use @ref dv_infer_wait_for_completion or
 * @ref dv_infer_wait_for_all_completion to collect results. Do not read output
 * blobs until the inference has reached @c DV_INFERENCE_STATUS_COMPLETED.
 * Memory for the inference object is allocated by the API and must be freed using
 * @ref dv_infer_free after reading the results.
 *
 * @param[in]  session       Session handle.
 * @param[in]  endpt         Endpoint or endpoint group handle to submit inference to. Although the
 *                           model already carries the endpoint group it was loaded on, this parameter
 *                           allows the caller to narrow or override the target to a specific endpoint
 *                           or sub-group within that group for this inference request.
 * @param[in]  model         Model handle for which inference is requested.
 * @param[in]  ip_array      Array of input blob descriptors, one per model input.
 * @param[in]  op_array      Array of output blob descriptors, one per model output.
 * @param[in]  enable_stats  Enable inference statistics collection (deprecated).
 * @param[out] inf_obj       Inference request object returned immediately after submission.
 * @return                   DV_SUCCESS on success, else error.
 * @note    Prefer @ref dv_infer_async_s which returns a detailed @ref dv_status_t
 *          instead of a plain status code.
 */
EXPORT
dv_status_code_t dv_infer_async(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, bool enable_stats, dv_infer_request_t **inf_obj);

/**
 * @brief Wrapper of dv_infer_async with detailed status.
 *
 * Use this instead of @ref dv_infer_async when you need a detailed status object
 * rather than a simple status code.
 *
 * @param[in]  session       Session handle.
 * @param[in]  endpt         Endpoint or endpoint group handle to submit inference to. Although the
 *                           model already carries the endpoint group it was loaded on, this parameter
 *                           allows the caller to narrow or override the target to a specific endpoint
 *                           or sub-group within that group for this inference request.
 * @param[in]  model         Model handle for which inference is requested.
 * @param[in]  ip_array      Array of input blob descriptors, one per model input.
 * @param[in]  op_array      Array of output blob descriptors, one per model output.
 * @param[in]  enable_stats  Enable inference statistics collection (deprecated).
 * @param[out] inf_obj       Inference request object returned immediately after submission.
 * @return                   Detailed status.
 */
EXPORT
dv_status_t dv_infer_async_s(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, bool enable_stats, dv_infer_request_t **inf_obj);

/**
 * @brief Waits until all inference requests in the provided list have changed status.
 *
 * Use this when you have submitted multiple async inference requests and need to wait
 * for all of them to complete before proceeding. If @p inf_obj_list is empty, blocks
 * until at least one inference request submitted on the session changes run status.
 * The API tracks status transitions per request and reports each status change only once.
 * For non-empty request lists, the caller is responsible for removing completed inference
 * objects from the list; completion status may be reported multiple times for requests
 * that remain in the list.
 *
 * @param[in]  session              Session handle.
 * @param[in]  inf_obj_list         Array of inference request handles to monitor.
 * @param[in]  inf_obj_count        Number of inference requests in @p inf_obj_list.
 * @param[in]  timeout              Maximum time in milliseconds to wait; pass -1 for default (60s).
 * @param[out] completed_inf_list   Array of inference handles that have completed.
 * @param[out] completed_inf_count  Number of completed inference handles returned.
 * @return                          DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_infer_wait_for_all_completion(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **completed_inf_list, int *completed_inf_count);

/**
 * @brief Waits until at least one inference request in the provided list has changed status.
 *
 * Use this in streaming or pipelined workloads where you want to process each completed
 * inference as soon as it finishes, without waiting for all requests to complete.
 * If you need all requests to finish before proceeding, use
 * @ref dv_infer_wait_for_all_completion instead. If @p inf_obj_list is empty, blocks
 * until at least one inference request submitted on the session changes run status.
 * The API tracks status transitions per request and reports each status change only once.
 * For non-empty request lists, the caller is responsible for removing completed inference
 * objects from the list; completion status may be reported multiple times for requests
 * that remain in the list.
 *
 * @param[in]  session        Session handle.
 * @param[in]  inf_obj_list   Array of inference request handles to monitor.
 * @param[in]  inf_obj_count  Number of inference requests in @p inf_obj_list.
 * @param[in]  timeout        Maximum time in milliseconds to wait; pass -1 for default (150s).
 * @param[out] inf_obj        Inference request handle for which the status has changed.
 * @return                    DV_SUCCESS on success, else error.
 * @note    Prefer @ref dv_infer_wait_for_completion_s which returns a detailed @ref dv_status_t
 *          instead of a plain status code.
 */
EXPORT
dv_status_code_t dv_infer_wait_for_completion(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **inf_obj);

/**
 * @brief Wrapper of dv_infer_wait_for_completion with detailed status.
 *
 * Use this instead of @ref dv_infer_wait_for_completion when you need a detailed
 * status object rather than a simple status code.
 *
 * @param[in]  session        Session handle.
 * @param[in]  inf_obj_list   Array of inference request handles to monitor.
 * @param[in]  inf_obj_count  Number of inference requests in @p inf_obj_list.
 * @param[in]  timeout        Maximum time in milliseconds to wait; pass -1 for default (150s).
 * @param[out] inf_obj        Inference request handle for which the status has changed.
 * @return                    Detailed status.
 */
EXPORT
dv_status_t dv_infer_wait_for_completion_s(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **inf_obj);

/**
 * @brief Returns the server-assigned request ID for a submitted inference request.
 *
 * Use this when debugging inference issues — the returned request ID matches the ID
 * printed in proxy logs when the proxy is started with the @c -t flag, allowing you
 * to correlate client-side requests with server-side log output. Call this after
 * submission but before freeing the inference handle.
 *
 * This API is not thread-safe. Calling this on a handle that has already been freed
 * via @ref dv_infer_free results in undefined behavior. If @p inf_obj is NULL or
 * @c inf_obj->handle is NULL, the function returns @c DV_INVALID_HOST_PTR and
 * @p req_id is not modified.
 *
 * @param[in]  inf_obj  Inference request handle.
 * @param[out] req_id   Server-assigned request ID for the inference.
 * @return              DV_SUCCESS on success; DV_INVALID_HOST_PTR if @p inf_obj or its handle is NULL.
 */
EXPORT
dv_status_code_t dv_infer_get_req_id(dv_infer_request_t *inf_obj, uint64_t *req_id);

/**
 * @brief Frees all resources associated with a completed inference request.
 *
 * Call this after reading all results from the inference object — output blobs,
 * statistics, and LLM info. Do not free an inference request that is still in flight;
 * wait for completion via @ref dv_infer_wait_for_completion first. After this call,
 * the handle is invalid and must not be accessed or passed to any API.
 * Failing to call this will leak memory in the client library.
 *
 * @param[in]  inf_obj  Inference request object to free.
 * @return              DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_infer_free(dv_infer_request_t *inf_obj);

/**
 * @brief Returns the number of in-flight inference requests for the session.
 *
 * Use this to implement backpressure in high-throughput pipelines — if the in-flight
 * count is high, delay submitting new requests to avoid overwhelming the proxy queue.
 * An in-flight request is one that has been submitted by the client library but for
 * which a response has not yet been received from the proxy server. This count includes
 * requests submitted by all threads sharing the same session.
 *
 * @param[in]  session  Session handle.
 * @param[out] count    Number of in-flight inference requests.
 * @return              DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_infer_get_inflight_count(dv_session_t *session, int *count);

/**
 * @brief Queries the busy state of a specific endpoint.
 *
 * Use this for lightweight load-balancing decisions — if the endpoint is busy,
 * consider routing the next inference request to a different endpoint. This is a
 * point-in-time snapshot and may not reflect the endpoint state by the time a new
 * inference is submitted. Do not use this in a tight polling loop; prefer
 * @ref dv_infer_wait_for_completion for waiting on specific inference results.
 * The caller must provide the memory for @p is_busy.
 *
 * @param[in]  session   Session handle.
 * @param[in]  ep        Endpoint handle to query.
 * @param[out] is_busy   Set to true if the endpoint is currently busy, false otherwise.
 * @return               DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_get_endpoint_busyness(dv_session_t *session, dv_endpoint_t *ep, bool *is_busy);

/**
 * @brief Submits a synchronous inference request with extended options and blocks until completion.
 *
 * Use this instead of @ref dv_infer_sync when you need LLM-specific inference control
 * (token counts, inference type) or fine-grained statistics collection via
 * @ref dv_infer_options_t. For standard CNN inference without special options, use
 * @ref dv_infer_sync instead. Memory for the inference object is allocated by the API
 * and must be freed using @ref dv_infer_free.
 *
 * @param[in]  session        Session handle.
 * @param[in]  endpt          Endpoint or endpoint group handle to submit inference to. Although the
 *                            model already carries the endpoint group it was loaded on, this parameter
 *                            allows the caller to narrow or override the target to a specific endpoint
 *                            or sub-group within that group for this inference request.
 * @param[in]  model          Model handle for which inference is requested.
 * @param[in]  ip_array       Array of input blob descriptors, one per model input.
 * @param[in]  op_array       Array of output blob descriptors, one per model output.
 * @param[in]  timeout        Maximum time in milliseconds to wait; pass -1 for default (60s).
 * @param[out] inf_obj        Inference request object returned on completion.
 * @param[in]  infer_options  Extended inference options controlling request behavior.
 * @return                    DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_infer_sync_with_options(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, int timeout, dv_infer_request_t **inf_obj, dv_infer_options_t *infer_options);

/**
 * @brief Submits an asynchronous inference request with extended options and returns immediately.
 *
 * Use this instead of @ref dv_infer_async when you need LLM-specific inference control
 * (token counts, inference type) or fine-grained statistics collection via
 * @ref dv_infer_options_t. For standard CNN inference without special options, use
 * @ref dv_infer_async instead. Memory for the inference object is allocated by the API
 * and must be freed using @ref dv_infer_free.
 *
 * @param[in]  session        Session handle.
 * @param[in]  endpt          Endpoint or endpoint group handle to submit inference to. Although the
 *                            model already carries the endpoint group it was loaded on, this parameter
 *                            allows the caller to narrow or override the target to a specific endpoint
 *                            or sub-group within that group for this inference request.
 * @param[in]  model          Model handle for which inference is requested.
 * @param[in]  ip_array       Array of input blob descriptors, one per model input.
 * @param[in]  op_array       Array of output blob descriptors, one per model output.
 * @param[out] inf_obj        Inference request object returned immediately after submission.
 * @param[in]  infer_options  Extended inference options controlling request behavior.
 * @return                    DV_SUCCESS on success, else error.
 * @note    Prefer @ref dv_infer_async_with_options_s which returns a detailed @ref dv_status_t
 *          instead of a plain status code.
 */
EXPORT
dv_status_code_t dv_infer_async_with_options(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, dv_infer_request_t **inf_obj, dv_infer_options_t *infer_options);

/**
 * @brief Wrapper of dv_infer_async_with_options with detailed status.
 *
 * Use this instead of @ref dv_infer_async_with_options when you need a detailed
 * status object rather than a simple status code.
 *
 * @param[in]  session        Session handle.
 * @param[in]  endpt          Endpoint or endpoint group handle to submit inference to. Although the
 *                            model already carries the endpoint group it was loaded on, this parameter
 *                            allows the caller to narrow or override the target to a specific endpoint
 *                            or sub-group within that group for this inference request.
 * @param[in]  model          Model handle for which inference is requested.
 * @param[in]  ip_array       Array of input blob descriptors, one per model input.
 * @param[in]  op_array       Array of output blob descriptors, one per model output.
 * @param[out] inf_obj        Inference request object returned immediately after submission.
 * @param[in]  infer_options  Extended inference options controlling request behavior.
 * @return                    Detailed status.
 */
EXPORT
dv_status_t dv_infer_async_with_options_s(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, dv_infer_request_t **inf_obj, dv_infer_options_t *infer_options);

/**
 * @brief Retrieves the output blobs for a specific output layer by name from a completed inference.
 *
 * Use this when a model has multiple output layers and you only need the output of a
 * specific layer by name. Must be called after the inference has completed successfully.
 * Do not call this on an in-flight or failed inference request. The returned blob array
 * is owned by the inference handle and must not be freed separately — freeing the
 * inference handle via @ref dv_infer_free will also free the memory pointed to by
 * @p op_blobs.
 *
 * @param[in]  inf_obj             Inference request handle from a successfully completed inference.
 * @param[in]  src_op_layer_name   Source graph output layer name as specified in the model's
 *                                 output layer parameters (@ref dv_model_output_param_t::src_graph_layer_name).
 * @param[out] op_blobs            Array of output blobs for the specified layer name.
 * @param[out] num_op_blobs        Number of blobs in the @p op_blobs array.
 * @return                         DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_fetch_outputs_by_layer_name(dv_infer_request_t *inf_obj, char *src_op_layer_name, dv_blob_t **op_blobs, int *num_op_blobs);

/**
 * @brief Loads a model from a file with extended options and transfers it to the specified endpoint.
 *
 * Use this as the preferred model load API when you need control over async loading,
 * disk caching, or explicit model type selection. Set @c options->async = true to
 * return immediately and poll for completion, which is useful when loading large LLM
 * models that take significant time to transfer. Set @c options->cache = true to cache
 * the model on disk for faster subsequent loads. Always set @c options->model_type
 * explicitly for LLM models — the default @c DV_MODEL_TYPE_ARA2_CNN is incorrect
 * for LLM workloads. If model loading fails on all devices represented by @p endpt,
 * the API returns an error.
 *
 * @param[in]  session          Session handle.
 * @param[in]  endpt            Endpoint or endpoint group handle to load the model onto.
 * @param[in]  model_file_path  Filesystem path to the compiled model file.
 * @param[out] model_handle     Model object returned on success.
 * @param[in]  options          Model load options controlling load behavior.
 * @return                      DV_SUCCESS if loaded successfully on all devices, DV_PARTIAL_SUCCESS if loaded on only some devices, DV_MODEL_LOAD_FAILURE if loading failed on all devices.
 * @note    Prefer @ref dv_model_load_from_file_with_options_s which returns a detailed @ref dv_status_t
 *          instead of a plain status code.
 */
EXPORT
dv_status_code_t dv_model_load_from_file_with_options(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, dv_model_t **model_handle, dv_model_load_options_t *options);

/**
 * @brief Wrapper of dv_model_load_from_file_with_options with detailed status.
 *
 * Use this instead of @ref dv_model_load_from_file_with_options when you need a
 * detailed status object rather than a simple status code.
 *
 * @param[in]  session          Session handle.
 * @param[in]  endpt            Endpoint or endpoint group handle to load the model onto.
 * @param[in]  model_file_path  Filesystem path to the compiled model file.
 * @param[out] model_handle     Model object returned on success.
 * @param[in]  options          Model load options controlling load behavior.
 * @return                      DV_SUCCESS if loaded successfully on all devices, DV_PARTIAL_SUCCESS if loaded on only some devices, DV_MODEL_LOAD_FAILURE if loading failed on all devices.
 */
EXPORT
dv_status_t dv_model_load_from_file_with_options_s(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, dv_model_t **model_handle, dv_model_load_options_t *options);

/**
 * @brief Loads a model from a blob with extended options and transfers it to the specified endpoint.
 *
 * Use this when the model is already in host memory and you need extended load control
 * such as async loading or explicit model type selection. Prefer this over
 * @ref dv_model_load_from_blob when loading LLM models or when async loading is needed.
 * If model loading fails on all devices represented by @p endpt, the API returns an error.
 *
 * @param[in]  session       Session handle.
 * @param[in]  endpt         Endpoint or endpoint group handle to load the model onto.
 * @param[in]  blob          Blob descriptor pointing to the model data in memory.
 * @param[in]  options       Model load options controlling load behavior.
 * @param[out] model_handle  Model object returned on success.
 * @return                   DV_SUCCESS if loaded successfully on all devices,
 *                           DV_PARTIAL_SUCCESS if loaded on only some devices,
 *                           Error Status if loading failed on all devices.
 * @note    Prefer @ref dv_model_load_from_blob_with_options_s which returns a detailed @ref dv_status_t
 *          instead of a plain status code.
 */
EXPORT
dv_status_code_t dv_model_load_from_blob_with_options(dv_session_t *session, dv_endpoint_t *endpt, dv_blob_t *blob, dv_model_t **model_handle, dv_model_load_options_t *options);

/**
 * @brief Wrapper of dv_model_load_from_blob_with_options with detailed status.
 *
 * Use this instead of @ref dv_model_load_from_blob_with_options when you need a
 * detailed status object rather than a simple status code.
 *
 * @param[in]  session       Session handle.
 * @param[in]  endpt         Endpoint or endpoint group handle to load the model onto.
 * @param[in]  blob          Blob descriptor pointing to the model data in memory.
 * @param[in]  options       Model load options controlling load behavior.
 * @param[out] model_handle  Model object returned on success.
 * @return                   Detailed status structure.
 */
EXPORT
dv_status_t dv_model_load_from_blob_with_options_s(dv_session_t *session, dv_endpoint_t *endpt, dv_blob_t *blob, dv_model_t **model_handle, dv_model_load_options_t *options);

/**
 * @brief Retrieves the current active version of each product component from the server.
 *
 * Use this at startup to verify that the client library version is compatible with
 * the running proxy and firmware. For the full set of versions supported by the proxy
 * (not just the currently active ones), use @ref dv_retrieve_version_details instead.
 * Memory for @p product_version is allocated by the API and must be freed using
 * @ref dv_free_version_details.
 *
 * @param[in]  session          Session handle.
 * @param[out] product_version  Array of product version structures returned by the server.
 * @param[out] product_count    Number of product version entries in the returned array.
 * @return                      DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_exchange_current_version_details(dv_session_t *session, dv_product_version_t **product_version, uint8_t *product_count);

/**
 * @brief Retrieves all supported versions for each product component from the server.
 *
 * Use this to determine the full compatibility matrix of the running proxy — useful
 * when managing deployments with multiple client library versions. To get only the
 * currently active versions, use @ref dv_exchange_current_version_details instead.
 * Memory for @p product_version is allocated by the API and must be freed using
 * @ref dv_free_version_details.
 *
 * @param[in]  session          Session handle.
 * @param[out] product_version  Array of product version structures returned by the server.
 * @param[out] product_count    Number of product version entries in the returned array.
 * @return                      DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_retrieve_version_details(dv_session_t *session, dv_product_version_t **product_version, uint8_t *product_count);

/**
 * @brief Retrieves the version of the currently running client library.
 *
 * Use this to log or validate the client library version at application startup,
 * or to include version information in bug reports. This does not require a session
 * and can be called before @ref dv_session_create_via_unix_socket. Populates the
 * provided version structure with the major, minor, patch, and patch_minor version
 * fields of the client library in use.
 *
 * @param[out] client_lib_version  Version structure to populate with client library version.
 * @return                         DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_get_client_lib_version(dv_version_t *client_lib_version);

/**
 * @brief Frees memory allocated by @ref dv_exchange_current_version_details or
 *        @ref dv_retrieve_version_details.
 *
 * Always call this after processing version details to avoid memory leaks.
 * Do not access @p product_version after this call.
 *
 * @param[in]  product_version  Pointer to the product version array to free.
 * @return                      DV_SUCCESS on success, else error.
 */
EXPORT
dv_status_code_t dv_free_version_details(dv_product_version_t *product_version);

#ifdef __cplusplus
}
#endif

#endif  // __DV_API_H__
