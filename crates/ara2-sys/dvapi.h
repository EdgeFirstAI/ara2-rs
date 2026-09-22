/*
 * Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
 * Kinara  Proprietary. This software is owned or controlled by Kinara and may only be used strictly in accordance with the applicable license terms.
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

struct dv_infer_request_handle;
typedef struct dv_infer_request_handle dv_infer_request_handle_t;

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
  DV_ENDPOINT_STATE_INIT = 0,             /**< endpoint is in init state */
  DV_ENDPOINT_STATE_IDLE = 1,             /**< endpoint is in idle state */
  DV_ENDPOINT_STATE_ACTIVE = 2,           /**< endpoint is in active state */
  DV_ENDPOINT_STATE_ACTIVE_SLOW = 3,      /**< endpoint is operating at reduced frequency*/
  DV_ENDPOINT_STATE_ACTIVE_BOOSTED = 4,   /**< endpoint is operating at reduced frequency*/
  DV_ENDPOINT_STATE_THERMAL_INACTIVE = 5, /**< endpoint is in thermal Inactive state */
  DV_ENDPOINT_STATE_THERMAL_UNKNOWN = 6,  /**< endpoint is in unown thermal state*/
  DV_ENDPOINT_STATE_INACTIVE = 7,         /**< endpoint is in Inactive state */
  DV_ENDPOINT_STATE_FAULT = 8,            /**< endpoint is in faulty state */
  DV_ENDPOINT_STATE_BAD_INTERFACE = 1001, /**< [unsupported] */
  DV_ENDPOINT_STATE_RECOVERY = 1003,      /**< [unsupported] */
  DV_ENDPOINT_STATE_DEAD = 1004,          /**< [unsupported] */
  DV_ENDPOINT_STATE_DRAIN = 1005,         /**< [unsupported] */
  DV_ENDPOINT_STATE_POWER_GATED = 1006,   /**< [unsupported] */
  DV_ENDPOINT_STATE_CLOSED = 1007,        /**< [unsupported] */
} dv_endpoint_state_t;

/// Endpoint Power State [currently unsupported]
typedef enum DV_ENDPOINT_POWER_STATE {
  DV_POWER_STATE_L0 = 0,  /**< endpoint is in L0 state */
  DV_POWER_STATE_L1 = 1,  /**< endpoint is in L1 state */
  DV_POWER_STATE_L1A = 2, /**< endpoint is in L1A state*/
  DV_POWER_STATE_L2 = 3,  /**< endpoint is in L2 state */
} dv_endpoint_power_state_t;

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

typedef enum DV_PRODUCT_TYPE {
  INVALID_PRODUCT = -1,
  PROXY,
  PCI_DRIVER,
  FIRMWARE,
  CNN_MODEL,
  LLM_MODEL,
  CLIENT_LIB,
  SYSAPI,
} dv_product_type_t;

typedef struct dv_version {
  uint8_t major;
  uint8_t minor;
  uint8_t patch;
  uint8_t patch_minor;
} dv_version_t;

typedef struct dv_product_version {
  dv_product_type_t product_type;
  dv_version_t product_version;
} dv_product_version_t;

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

// DV Endpoint chip information
typedef struct dv_endpoint_chip_info {
  char *id;                    /**< dv chip id */
  char *rev;                   /**< dv chip revision */
  int control_processor_count; /**< dv chip control processor count */
  int neural_processor_count;  /**< dv chip neural processor count */
  uint32_t l2_memory_size;     /**< dv chip internal L2 memory size in bytes */
} dv_endpoint_chip_info_t;

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
    char *pcie_dir; /**< \since r5.3 path to the pcie sysfs entry for a PCIE
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

// Inference queue statistics per endpoint
typedef struct dv_inference_queue_statistics {
  int occupancy_count; /**< Number of inference queue slots occupied with inference request for the endpoint */
  int length;          /**< length of the inference queue for the endpoint */
  float wait_time;     /**< waiting time in mili secs for the new inference request to get picked up by endpoint */
} dv_inference_queue_statistics_t;

// Model statistics per endpoint
typedef struct dv_model_statistics {
  uint32_t model;                                    /**< model handle */
  uint32_t active_input_tensors_count;               /**< number for active model input tensor(s) present in an endpoint */
  uint32_t active_output_tensors_count;              /**< number for active model output tensor(s) present in an endpoint */
  uint32_t active_inferences_count;                  /**< number for active model inference request queued in an endpoint */
  uint32_t model_total_dram_occupancy_size;          /**< total endpoint dram occupancy in bytes by model artefacts */
  uint32_t model_total_input_tensor_occupancy_size;  /**< total endpoint dram occupancy in bytes by model input tensors */
  uint32_t model_total_output_tensor_occupancy_size; /**< total endpoint dram occupancy in bytes by model output tensors */
  dv_model_handle_t *model_handle;                   /**< \since r5.3 void* handle which can be compared to `handle` member of dv_model_t */
} dv_model_statistics_t;

// Endpoint statistics
typedef struct dv_endpoint_stats {
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
  bool is_signed;            /**< input range ((-128) - (128)) / (0 - 255) */
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
} dv_model_output_param_t;

// Power and performance reported by DVNC(Kinara Network Compiler)
typedef struct dv_compiler_statistics {
  char *config_name;   /**< DV1 config name, governed on ep system core clock */
  float cycles;        /**< total cycles estimated by compiler */
  float ips;           /**< inference per seconds estimated by compiler */
  float ddr_bandwidth; /**< ep dram estimated by compiler */
} dv_compiler_statistics_t;

typedef enum DV_MODEL_TYPE {
  DV_MODEL_TYPE_ARA1_CNN = 0,
  DV_MODEL_TYPE_ARA2_CNN = 1,
  DV_MODEL_TYPE_ARA2_LLM = 2,         // this is for dynamic quant v1 qwen models
  DV_MODEL_TYPE_ARA2_LLM_DYN_V2 = 3,  // latest llm models [except for dynmaic quant v1 models]
  // DV_MODEL_TYPE_ARA2_LLM_DYN_V1_LLAMA = 5, // redundant, can be removed
  // DV_MODEL_TYPE_ARA2_LLM_STATIC_QWEN = 6 // redundant, can be removed
} dv_model_type_t;

// arguments to be passed to dv_model_load_*_with_options APIs \since r6.0
typedef struct dv_model_options {
  char *model_name;                   /**< model name */
  dv_model_priority_level_t priority; /**< priority of the model [unused] */
  bool cache;                         /**< if true, the model is cached on disk */
  bool async;                         /**< if true, the model load API immediately return \see
                                         dv_model_load_wait_for_completion */
  dv_model_type_t model_type;         /**< specify the model type, if not specified it will be  DV_MODEL_TYPE_ARA2_CNN*/
} dv_model_load_options_t;

typedef enum DV_INFER_TYPE {
  DV_INFER_TYPE_ARA1_CNN = 0,
  DV_INFER_TYPE_ARA2_CNN = 1,
  DV_INFER_TYPE_LLM_PROMPT_PROCESSING = 2,
  DV_INFER_TYPE_LLM_FOLLOWUP_PROMPT_PROCESSING = 3,
  DV_INFER_TYPE_LLM_TOKEN_GENERATION = 4,

} dv_infer_type_t;

typedef struct dv_infer_options {
  // bool async;
  bool enable_stats;
  // uint64_t timeout_ms;
  dv_infer_type_t infer_type; /**< specify the model type, if not specified it will be  DV_MODEL_TYPE_ARA2_CNN*/
  uint64_t active_tokens;
  uint32_t valid_tokens;
  uint32_t tokens_to_skip;  // number of image, video tokens in the prompt.
} dv_infer_options_t;

// fixme amit :: can be changed to uint64_t, we should add a field for model type [dynamic/specD etc.]
// fixme amit :: add a field num_outputs
typedef struct dv_model_llm_params {
  uint32_t vocab_size;
  uint32_t embedding_size;
  uint32_t input_precision;
  uint32_t output_precision;
  uint32_t max_num_tokens;
  uint32_t is_dynamic;
  uint32_t num_inputs;
  uint32_t pad_token_id;
  uint32_t eos_token_id;
  uint32_t bos_token_id;
  uint64_t embedding_lookup_addr;
  uint64_t embedding_lookup_scale_addr;
} dv_model_llm_params_t;

typedef struct {
  uint32_t top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  uint32_t target_token_post_mcp;
  uint32_t target_token_pre_mcp;
  uint32_t target_prompt_post_mcp;
  uint32_t target_prompt_pre_mcp;
  uint32_t draft_token_post_mcp;
  uint32_t draft_token_pre_mcp;
  uint32_t draft_prompt_post_mcp;
  uint32_t draft_prompt_pre_mcp;
} dv_llm_cfg_upd_req_t;

// Model object
typedef struct dv_model {
  dv_model_handle_t *handle;                   /**< private handle, managed by client library */
  dv_session_t *session;                       /**< session handle on which model is loaded */
  dv_endpoint_t *endpoint;                     /**< endpoint handle on which model is loaded */
  dv_version_t version;                        /**< model version as generated by compiler */
  char *name;                                  /**< model name provided by user */
  dv_model_type_t model_type;                  /**< model type */
  char *internal_name;                         /**< internal model name as generated by compiler */
  int num_inputs;                              /**< number of inputs needed by model */
  int num_outputs;                             /**< number of output produced by model */
  dv_model_priority_level_t priority;          /**< model priority as set by user [unused]*/
  dv_model_input_param_t *input_param;         /**< list of input params */
  dv_model_output_param_t *output_param;       /**< list of output params */
  dv_model_llm_params_t *llm_params;           /**< list of llm params, valid if model is llm model*/
  int num_compiler_config;                     /**< [unsupported] */
  dv_compiler_statistics_t *compiler_stats;    /**< [unsupported] */
  dv_model_load_options_t *model_load_options; /**< [unsupported]*/
  bool cp_layer;                               /**< [unsupported] */
} dv_model_t;

// Inference statistics
typedef struct dv_infer_statistics {
  int ep_hw_sys_clk;                     /**< ep hardware system core clock in MHz */
  int ep_hw_nnp_clk;                     /**< ep hardware external nnp clock in MHz */
  int ep_hw_sbp_clk;                     /**< ep hardware external sbp clock in MHz */
  int ep_hw_dram_clk;                    /**< ep hardware external dram clock in MHz */
  uint32_t ep_hw_total_inference_cycles; /**< total cycles taken to compute
                                            inference in hardware, including
                                            floating point computation */
  uint32_t ep_hw_fp_cycles;              /**< cycles taken to compute floating point
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

typedef struct dv_infer_llm_info {
  uint32_t llm_infer_resp_num_valid_tokens;
} dv_infer_llm_info_t;

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
  dv_infer_llm_info_t *llm_infer_info;
} dv_infer_request_t;

/********************************** DV Client APIs
 * *************************************************/

/** Stringify status code
 * \param[in]   status_code : status code
 * \param[out]  status      : status code as string
 */
EXPORT
const char *dv_stringify_status_code(dv_status_code_t status_code);

/********************************** DV Session APIs
 * *************************************************/

/** Create a session to the server using unix file socket
 * Returns DV_SUCCESS on success, else error
 * \param[in]   socket_file_path : server unix socket path
 * \param[out]  session          : session handle returned by api
 */
EXPORT
dv_status_code_t dv_session_create_via_unix_socket(const char *socket_file_path, dv_session_t **session);

/** Create a session to the server using windows named pipe
 * Returns DV_SUCCESS on success, else error
 * \param[in]   named_pipe       : server pipe
 * \param[out]  session          : session handle returned by api
 */
EXPORT
dv_status_code_t dv_session_create_via_named_pipe(const char *named_pipe, dv_session_t **session);

/** Create a session to the server using TCP/IPv4 sockets
 * Returns DV_SUCCESS on success, else error
 * \param[in]  tcp_ip_addr      : server ipv4 address
 * \param[in]  port             : server port
 * \param[out] session          : session handle returned by api
 */
EXPORT
dv_status_code_t dv_session_create_via_tcp_ipv4_socket(const char *tcp_ip_addr, int port, dv_session_t **session);

/** Close the session to the server
 * Returns DV_SUCCESS on success, else error
 * \param[in]  session  : session handle
 */
EXPORT
dv_status_code_t dv_session_close(dv_session_t *session);

/********************************** DV SharedFD API
 * ***************************************/
/** Register a file descriptor with the server
 * Returns DV_SUCCESS on success, else error
 * \param[in]  session      : session object
 * \param[in]  fd           : file fd to share
 * \param[in]  size         : size to mmap
 * \param[in]  offset       : file offset
 * \param[in]  fd_type      : fd type to provide hint to server (reserved)
 * \param[out] shm_desc     : shared memory descriptor
 */
EXPORT
dv_status_code_t dv_shmfd_register(dv_session_t *session, int fd, uint32_t size, uint32_t offset, int fd_type, dv_shm_descriptor_t **shm_desc);

/** Un-register shared file with the server
 * Returns DV_SUCCESS on success, else error
 * \param[in]  shm_desc     : shared buffer descriptor
 */
EXPORT
dv_status_code_t dv_shmfd_unregister(dv_shm_descriptor_t *shm_desc);

/********************************** DV Endpoint APIs
 * *************************************************/

/** Get the list of endpoints connected to the server
 * Endpoint list memory is allocated by the api. It also returns the endpoint
 * count connected to server Returns DV_SUCCESS on success, else error
 * \param[in]   session         : session handle
 * \param[out]  ep_list         : connected endpoint list returned by server
 * \param[out]  ep_count        : number of endpoint connected to server
 */
EXPORT
dv_status_code_t dv_endpoint_get_list(dv_session_t *session, dv_endpoint_t **ep_list, int *ep_count);

/** Get the default endpoint group supported by server.
 * Type of default group supported is defined in enum DV_ENDPOINT_DEFAULT_GROUP.
 * API allocates the memory for the endpoint configuration of the group.
 * Application needs to free the memory using dv_endpoint_free_group() Returns
 * DV_SUCCESS on success, else error \param[in]   session      : session handle
 * \param[in]   grp          : enum as defined in enum DV_ENDPOINT_DEFAULT_GROUP
 * \param[out]  ep_grp       : endpoint group handle
 */
EXPORT
dv_status_code_t dv_endpoint_get_default_group(dv_session_t *session, dv_endpoint_default_group_t grp, dv_endpoint_t **ep_grp);

/** Create custom endpoint group from the endpoints connected to the server
 * API allocates the memory for the endpoint configuration of the group.
 * Application needs to free the memory using dv_endpoint_free_group() Returns
 * DV_SUCCESS on success, else error \param[in]   session      : session handle
 * \param[in]   ep_list      : endpoint list for which custom group needs to be
 * created \param[in]   ep_count     : number of endpoint in the list
 * \param[out]  ep_grp       : endpoint group handle
 */
EXPORT
dv_status_code_t dv_endpoint_create_group(dv_session_t *session, dv_endpoint_t **ep_list, int ep_count, dv_endpoint_t **ep_grp);

/** Free the default/custom endpoint group
 * Returns DV_SUCCESS on success, else error
 * \param[in]   session      : session handle
 * \param[out]  ep_list      : connected endpoint list returned by server
 * \param[out]  ep_count     : number of endpoint connected to server
 */
EXPORT
dv_status_code_t dv_endpoint_free_group(dv_endpoint_t *ep_grp);

/** Detects endpoint state
 * \param[in]   session     : session handle
 * \param[in]   ep          : endpoint handle
 * \param[out]  state       : endpoint state
 */
EXPORT
dv_status_code_t dv_endpoint_check_status(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_state_t *state);

/********************************** DV Endpoint stats APIs
 * *************************************************/

/** Get the dram usage statistics for the endpoint/endpoint group
 * Server provides the snapshot of the endpoint dram usage at the time of
 * request. Passing NULL to endpoint handle will provide dram statistics for all
 * the endpoint connected to the server Memory for the statistics is allocated
 * by API and dv_endpoint_free_dram_statistics() is needed to free up the memory
 * Returns DV_SUCCESS on success, else error
 * \param[in]   session         : session handle
 * \param[in]   ep              : endpoint/endpoint group handle for particular
 * endpoint or endpoint group or NULL for all endpoints \param[out]
 * ep_dram_stats   : list of endpoint dram statistics \param[out]  ep_count :
 * number of endpoint for which dram statistics is provided
 */
EXPORT
dv_status_code_t dv_endpoint_get_dram_statistics(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_dram_statistics_t **ep_dram_stats, int *ep_count);

/** Free the endpoint dram stats memory allocated in
 * dv_endpoint_get_dram_statistics() Returns DV_SUCCESS on success, else error
 * \param[in] ep_dram_stats : endpoint dram statistics to free
 * \param[in] count         : number of items to free
 */
EXPORT
dv_status_code_t dv_endpoint_free_dram_statistics(dv_endpoint_dram_statistics_t *ep_dram_stats, int count);

/** Get all the statistics for the endpoint/endpoint group
 * Server provides the snapshot of the endpoint statistics at the time of
 * request. Passing NULL to endpoint handle will provide statistics for all the
 * endpoint connected to the server Memory for the statistics is allocated by
 * API and dv_endpoint_free_statistics() is needed to free up the memory Returns
 * DV_SUCCESS on success, else error
 *
 * Notes on usage
 * -------------
 * `dv_model_statistics_t` member of `dv_endpoint_statistics_t` contains 2
 * fields - uint32_t model and void* model_handle. This API will return stats of
 * all models loaded across all clients - the `uint32_t model` will always hold
 * a value and is an server generated id. The value of the `void* model_handle`
 * member can be compared to the `handle` member of `dv_model_t` to map
 * `dv_model_t` objects to their corresponding `dv_model_statistics_t` objects.
 *
 * \param[in]   session     : session handle
 * \param[in]   ep          : endpoint/endpoint group handle for particular
 * endpoint or endpoint group or NULL for all endpoints \param[out]  ep_stats :
 * list of endpoint statistics \param[out]  ep_count    : number of endpoint for
 * which statistics is provided
 */
EXPORT
dv_status_code_t dv_endpoint_get_statistics(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_statistics_t **ep_stats, int *ep_count);

/** Free the endpoint stats memory allocated in dv_endpoint_get_statistics()
 * Returns DV_SUCCESS on success, else error
 * \param[in] ep_stats  : endpoint statistics list to free
 * \param[in] count     : number of items to free
 */
EXPORT
dv_status_code_t dv_endpoint_free_statistics(dv_endpoint_statistics_t *ep_stats, int count);

/********************************** DV Model APIs
 * *************************************************/

/** Creates a model object and load model contents from file and transfer it to
 * endpoint Model object contains the model handle and model parameters.
 * `dv_model_get_loaded_endpoint_list` returns the list of endpoints (individual
 * device dv_endpoint_t handles) on which the model was successfully loaded.
 *
 * If model load fails on all individual devices representing `endpt`, the API
 * call will error out.
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   session          : session handle
 * \param[in]   endpt            : endpoint handle
 * \param[in]   model_file_path  : model file path
 * \param[in]   priority         : model priority [unused]
 * \param[in]   model_name       : model name [unused]
 * \param[out]  model_handle     : model handle returned by server
 */
EXPORT
dv_status_code_t dv_model_load_from_file(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);

/** Creates a model object and load model contents from blob and transfer it to
 * endpoint Model object contains the model handle and model parameters.
 * `dv_model_get_loaded_endpoint_list` returns the list of endpoints (individual
 * device dv_endpoint_t handles) on which the model was successfully loaded.
 *
 * If model load fails on all individual devices representing `endpt`, the API
 * call will error out.
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   session          : session handle
 * \param[in]   endpt            : endpoint handle
 * \param[in]   model_blob       : model blob
 * \param[in]   priority         : model priority [unused]
 * \param[in]   model_name       : model name [unused]
 * \param[out]  model_handle     : model handle returned by server
 */
EXPORT
dv_status_code_t dv_model_load_from_blob(dv_session_t *session, dv_endpoint_t *endpt, dv_blob_t *model_blob, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);

/** Unloads a model from the endpoint
 * Returns DV_SUCCESS on success, else error
 * \param[in]  model    : model handle
 */
EXPORT
dv_status_code_t dv_model_unload(dv_model_t *model);

/** Get the model parameters from model file without loading the model to
 * endpoint(s) API allocates memory for the parameters. Application can free it
 * by calling dv_model_free_paramaters This is a utility function, useful to get
 * model parameters without loading the model to endpoint(s). The model object
 * returned by the api cannot be used Returns DV_SUCCESS on success, else error
 * \param[in]  model_file_path  : model file path
 * \param[in]  model            : model parameters
 */
EXPORT
dv_status_code_t dv_model_get_parameters_from_file(const char *model_file_path, dv_model_t **model);

/** Get the model parameters from model blob without loading the model onto
 * endpoint(s) API allocates memory for the parameters. Application can free it
 * by calling dv_model_free_paramaters This is a utility function, useful to get
 * model parameters without loading the model to endpoint(s) Returns DV_SUCCESS
 * on success, else error \param[in]  model_blob  : model blob \param[in]  model
 * : model parameters
 */
EXPORT
dv_status_code_t dv_model_get_parameters_from_blob(dv_blob_t *model_blob, dv_model_t **model);

/** Free the memory allocations for model parameter made by
 * dv_model_get_parameters_* call Returns DV_SUCCESS on success, else error
 * \param[in]  model    : model handle
 */
EXPORT
dv_status_code_t dv_model_free_parameters(dv_model_t *model);

/** Pass pre/post processing flags and params to proxy/mcp
 * \param[in]  session      : session handle
 * \param[in]  endpt        : endpoint handle
 * \param[in] model             : model handle // currently unused by proxy
 * \param[in] llm_cfg_update    : dv_llm_cfg_upd_req_t type representing pre/post processing flags and params
 */
EXPORT
dv_status_code_t dv_model_set_llm_cfg_params(dv_session_t *session, dv_endpoint_t *ep, dv_model_t *model, dv_llm_cfg_upd_req_t *llm_cfg_update);
/********************************** DV Inference APIs
 * *************************************************/

/** Submit an synchronous inference for a model to an endpoint(group)
 * This is a blocking call, it returns when inference is completed
 * (success/error) or a timeout has occurred Input and output blobs should be
 * provided as per input and output parameter provided by model Inference object
 * memory will be allocated by API and can be freed upon calling dv_infer_free()
 * Returns DV_SUCCESS on success, else error
 * \param[in]  session      : session handle
 * \param[in]  endpt        : endpoint handle
 * \param[in]  model        : model handle
 * \param[in]  ip_array     : an array of input blob descriptor
 * \param[in]  op_array     : an array of output blob descriptor
 * \param[in]  timeout      : maximum time in mili seconds to wait for request
 * to complete \param[in]  enable_stats : enable stats for the inference request
 * (deprecated) \param[out] inf_obj      : inference object
 */
EXPORT
dv_status_code_t dv_infer_sync(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, int timeout, bool enable_stats, dv_infer_request_t **inf_obj);

/** Submit an asynchronous inference for a model to an endpoint(group)
 * Server returns a inference request id which can be used to get the completion
 * status or wait upon for compleetion after the call. Input and output blobs
 * should be provided as per input and output parameter provided by model
 * Inference object memory will be allocated by API and can be freed upon
 * calling dv_infer_free() Returns DV_SUCCESS on success, else error \param[in]
 * session      : session handle \param[in]  endpt        : endpoint handle
 * \param[in]  model        : model handle
 * \param[in]  ip_array     : an array of input blob descriptor
 * \param[in]  op_array     : an array of output blob descriptor
 * \param[in]  enable_stats : enable stats for the inference request
 * (deprecated) \param[out] inf_obj      : inference object
 */
EXPORT
dv_status_code_t dv_infer_async(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, bool enable_stats, dv_infer_request_t **inf_obj);

/** Monitor multiple inference request object and wait until, all of the
 * request updates/changes run status If the list is empty it waits until,
 * atleast one inference request submitted in the session changes run status and
 * API keeps track of status change for a inference request and run status is
 * reported only once.
 *
 * For non-empty request list, application needs to remove the completed
 * inference object from the list. Request completion status can be reported
 * multiple times.
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]  session      : session handle
 * \param[in]  inf_id_list  : inference request object list
 * \param[in]  inf_id_count : number of inference request to monitor
 * \param[in]  timeout      : maximum time in mili seconds to wait for inference
 * request to change status; defaults to 60 seconds in case timeout is passed as
 * -1
 */

EXPORT
dv_status_code_t dv_infer_wait_for_all_completion(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **completed_inf_list, int *completed_inf_count);

/** Monitor multiple inference request object and wait until, atleast one of the
 * request updates/changes run status If the list is empty it waits until,
 * atleast one inference request submitted in the session changes run status and
 * API keeps track of status change for a inference request and run status is
 * reported only once.
 *
 * For non-empty request list, application needs to remove the completed
 * inference object from the list. Request completion status can be reported
 * multiple times.
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]  session      : session handle
 * \param[in]  inf_id_list  : inference request object list
 * \param[in]  inf_id_count : number of inference request to monitor
 * \param[in]  timeout      : maximum time in mili seconds to wait for inference
 * request to change status; defaults to 60 seconds in case timeout is passed as
 * -1 \param[out] inf_obj      : inference id for which the state is changed
 */
EXPORT
dv_status_code_t dv_infer_wait_for_completion(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **inf_obj);

/**
 * Returns the request id of an inference that is sent by the client.
 * The request id returned will be the same printed in logs if the proxy is
 * passed the `-t` flag.
 *
 * If the inf_obj is null or if inf_obj->handle is null, DV_INVALID_HOST_PTR is
 * returned. The req_id argument is not set to any value in this case.
 *
 * If return value is DV_SUCCESS req_id will contain the request id of the
 * inference request sent to kinara inference proxy.
 *
 * This api is not thread safe. Calling this API on an inference handle that has
 * already been `dv_infer_free`'d is incorrect and will result in undefined
 * behaviour.
 *
 * \param[in]  inf_obj   : inference request handle
 * \param[out] req_id    : request id of the inference request
 * \since r6.1.1
 */
EXPORT
dv_status_code_t dv_infer_get_req_id(dv_infer_request_t *inf_obj, uint64_t *req_id);

/** Finish the life cycle for the inference request.
 * API will free up the associated memory for the inference request and will no
 * longer be accessible. After completion, any operation on the inference
 * request will be invalid. Returns DV_SUCCESS on success, else error \param[in]
 * inf_obj      : inference object
 */
EXPORT
dv_status_code_t dv_infer_free(dv_infer_request_t *inf_obj);

/** Returns the number of inflight inference requests for the session object for
 * which the client library has not recieved a response from the proxy server.
 * \param[in]  session      : session handle
 * \param[out] count        : number of inference requests in flight
 * \since r5.3
 * */
EXPORT
dv_status_code_t dv_infer_get_inflight_count(dv_session_t *session, int *count);

/** Get the busyness for the endpoint
 * Server provides busyness of any one endpoint at the time of request
 * Memory for is_busy bool variable should be provided by client app
 * Returns DV_SUCCESS on success, else error
 * Notes on usage
 * -------------
 *
 * \param[in]   session     : session handle
 * \param[in]   ep          : endpoint handle for particular endpoint
 * \param[out]  is_busy    : will return true if endpoint is busy else false
 */
EXPORT
dv_status_code_t dv_get_endpoint_busyness(dv_session_t *session, dv_endpoint_t *ep, bool *is_busy);

EXPORT
dv_status_code_t dv_infer_sync_with_options(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, int timeout, dv_infer_request_t **inf_obj, dv_infer_options_t *infer_options);

EXPORT
dv_status_code_t dv_infer_async_with_options(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, dv_infer_request_t **inf_obj, dv_infer_options_t *infer_options);

/**
 * Get output blob for given output layer names
 *
 * API is available from r1.2 release onwards
 * Returns an array of output blobs for a given layer name
 * Api should be called by passing infer handle for a successfully run inference
 * Freeing infer handle will free memory pointed by output_blobs
 * \param[in] inf_obj infer request handle received after running inference successfully
 * \param[in] src_op_layer_name source graph output layer name, as specified in model output layer params struct
 * \param[out] op_blobs array of output blobs for given src_op_layer_name
 * \param[out] num_op_blobs number of output blobs in op_blobs
 */
EXPORT
dv_status_code_t dv_fetch_outputs_by_layer_name(dv_infer_request_t *inf_obj, char *src_op_layer_name, dv_blob_t **op_blobs, int *num_op_blobs);

/** Creates a model object and load model contents from blob and transfer it to
 * endpoint Model object contains the model handle and model parameters.
 * `dv_model_get_loaded_endpoint_list` returns the list of endpoints (individual
 * device dv_endpoint_t handles) on which the model was successfully loaded.
 *
 * If model load fails on all individual devices representing `endpt`, the API
 * call will error out.
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   session          : session handle
 * \param[in]   endpt            : endpoint handle
 * \param[in]   model_file_path  : path to model file
 * \param[out]  model_handle     : model handle returned by server
 * \param[in]   options          : model load options
 * \since       r6.0
 */
EXPORT
dv_status_code_t dv_model_load_from_file_with_options(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, dv_model_t **model_handle, dv_model_load_options_t *options);

/** Creates a model object and load model contents from blob and transfer it to
 * endpoint Model object contains the model handle and model parameters.
 * `dv_model_get_loaded_endpoint_list` returns the list of endpoints (individual
 * device dv_endpoint_t handles) on which the model was successfully loaded.
 *
 * If model load fails on all individual devices representing `endpt`, the API
 * call will error out.
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   session          : session handle
 * \param[in]   endpt            : endpoint handle
 * \param[in]   model_blob       : dv_blob_t of the model
 * \param[in]   options          : model load options
 * \param[out]  model_handle     : model handle returned by server
 * \since       r6.0
 */
EXPORT
dv_status_code_t dv_model_load_from_blob_with_options(dv_session_t *session, dv_endpoint_t *endpt, dv_blob_t *blob, dv_model_t **model_handle, dv_model_load_options_t *options);

/** api gives current version details of
 *model, driver, proxy, firmware version to client
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]    session             : session handle
 * \param[out]   product_version     : product versions
 * \param[out]   product_count       : number of product version sent
 */
EXPORT
dv_status_code_t dv_exchange_current_version_details(dv_session_t *session, dv_product_version_t **product_version, uint8_t *product_count);

/** api gives all supported version details of
 *model, driver, proxy, firmware version to client
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   product_version        : list of versions
 */

EXPORT
dv_status_code_t dv_retrieve_version_details(dv_session_t *session, dv_product_version_t **product_version, uint8_t *product_count);

/** api gives current running version of client library
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   client_lib_version        : fill struct with client library version
 */
EXPORT
dv_status_code_t dv_get_client_lib_version(dv_version_t *client_lib_version);

/** api deletes memory created by dv_exchange_current_version_details or dv_retrieve_version_details apis
 *
 * Returns DV_SUCCESS on success, else error
 * \param[in]   product_version        : list of versions
 */
EXPORT
dv_status_code_t dv_free_version_details(dv_product_version_t *product_version);

#ifdef __cplusplus
}
#endif

#endif  // __DV_API_H__
