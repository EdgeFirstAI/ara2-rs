/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * libaraclient DVAPI declarations shared by every DVAPI version ara2
 * supports (1.1, 1.3 and 3.0): handles, enums, the structs whose layout is
 * identical in all three, and the functions whose signature is.
 *
 * Structs whose layout differs between versions appear here only as
 * incomplete types. dvapi_v1_1.h, dvapi_v1_3.h and dvapi_v3_0.h complete
 * them, and must refer to them by struct tag: the _t typedefs below name
 * the incomplete type, which bindgen emits as a zero-sized opaque struct.
 *
 * tests/verify_sdk.rs checks every declaration against an SDK's own
 * headers and relies on two conventions: every enumerator carries an
 * explicit value, and every function prototype sits on one line.
 */
#ifndef ARA2_DVAPI_COMMON_H
#define ARA2_DVAPI_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

/* Only the codes ara2 names. Every code is named at runtime by
 * dv_stringify_status_code. */
typedef enum dv_status_code {
  DV_SUCCESS = 0,
} dv_status_code_t;

typedef enum DV_BLOB_TYPE {
  DV_BLOB_TYPE_RAW_POINTER = 0,
  DV_BLOB_TYPE_SHM_DESCRIPTOR = 1,
  DV_BLOB_TYPE_FD = 2,
} dv_blob_type_t;

typedef enum DV_PRODUCT_TYPE {
  INVALID_PRODUCT = -1,
  PROXY = 0,
  PCI_DRIVER = 1,
  FIRMWARE = 2,
  CNN_MODEL = 3,
  LLM_MODEL = 4,
  CLIENT_LIB = 5,
  SYSAPI = 6,
} dv_product_type_t;

/* Values 4 and 5 mean different things in 1.1 and 1.3+, so they are
 * declared per version. */
typedef enum DV_ENDPOINT_STATE {
  DV_ENDPOINT_STATE_INIT = 0,
  DV_ENDPOINT_STATE_IDLE = 1,
  DV_ENDPOINT_STATE_ACTIVE = 2,
  DV_ENDPOINT_STATE_ACTIVE_SLOW = 3,
  DV_ENDPOINT_STATE_THERMAL_UNKNOWN = 6,
  DV_ENDPOINT_STATE_INACTIVE = 7,
  DV_ENDPOINT_STATE_FAULT = 8,
} dv_endpoint_state_t;

/* 3.0 adds values 4 to 9, declared in dvapi_v3_0.h. */
typedef enum DV_ENDPOINT_POWER_STATE {
  DV_POWER_STATE_L0 = 0,
  DV_POWER_STATE_L1 = 1,
  DV_POWER_STATE_L1A = 2,
  DV_POWER_STATE_L2 = 3,
} dv_endpoint_power_state_t;

typedef enum DV_ENDPOINT_HOST_INTERFACE {
  DV_ENDPOINT_HOST_INTERFACE_PCIE = 1,
  DV_ENDPOINT_HOST_INTERFACE_USB = 2,
} dv_endpoint_host_interface_t;

typedef enum DV_ENDPOINT_GROUP_TYPE {
  DV_ENDPOINT_GROUP_TYPE_NONE = 0,
  DV_ENDPOINT_GROUP_TYPE_ALL = 1,
  DV_ENDPOINT_GROUP_TYPE_PCIE = 2,
  DV_ENDPOINT_GROUP_TYPE_USB = 3,
  DV_ENDPOINT_GROUP_TYPE_CUSTOM = 4,
} dv_endpoint_group_type_t;

typedef enum DV_MODEL_TYPE {
  DV_MODEL_TYPE_ARA1_CNN = 0,
  DV_MODEL_TYPE_ARA2_CNN = 1,
  DV_MODEL_TYPE_ARA2_LLM = 2,
  DV_MODEL_TYPE_ARA2_LLM_DYN_V2 = 3,
} dv_model_type_t;

typedef enum DV_MODEL_PRIORITY_LEVEL {
  DV_MODEL_PRIORITY_LEVEL_LOW = 0,
  DV_MODEL_PRIORITY_LEVEL_MEDIUM = 1,
  DV_MODEL_PRIORITY_LEVEL_DEFAULT = 1,
  DV_MODEL_PRIORITY_LEVEL_HIGH = 2,
} dv_model_priority_level_t;

typedef enum DV_INFERENCE_STATUS {
  DV_INFERENCE_STATUS_QUEUED = 0,
  DV_INFERENCE_STATUS_RUNNING = 1,
  DV_INFERENCE_STATUS_COMPLETED = 2,
  DV_INFERENCE_STATUS_FAILED = 4,
  DV_INFERENCE_STATUS_UNKNOWN = 5,
} dv_inference_status_t;

typedef enum DV_LAYER_OUTPUT_TYPE {
  DV_LAYER_OUTPUT_TYPE_CLASSIFICATION = 0,
  DV_LAYER_OUTPUT_TYPE_DETECTION = 1,
  DV_LAYER_OUTPUT_TYPE_SEMANTIC_SEGMENTATION = 2,
  DV_LAYER_OUTPUT_TYPE_RAW = 3,
} dv_layer_output_type_t;

/* Handles ara2 only passes through. */
typedef struct dv_session dv_session_t;
typedef struct dv_model_handle dv_model_handle_t;
typedef struct dv_infer_request_handle dv_infer_request_handle_t;
typedef struct dv_model_llm_params dv_model_llm_params_t;
typedef struct dv_compiler_statistics dv_compiler_statistics_t;
typedef struct dv_model_load_options dv_model_load_options_t;

/* Completed per version. */
typedef struct dv_endpoint_dram_statistics dv_endpoint_dram_statistics_t;
typedef struct dv_endpoint_statistics dv_endpoint_statistics_t;
typedef struct dv_model_output_param dv_model_output_param_t;
typedef struct dv_model dv_model_t;

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

typedef struct dv_blob {
  void *handle;
  uint64_t offset;
  uint64_t size;
  dv_blob_type_t blob_type;
} dv_blob_t;

typedef struct dv_shm_descriptor {
  void *handle;
  dv_session_t *session;
  int fd;
  size_t size;
  size_t offset;
  int fd_type;
} dv_shm_descriptor_t;

typedef struct dv_endpoint_chip_info {
  char *id;
  char *rev;
  int control_processor_count;
  int neural_processor_count;
  uint32_t l2_memory_size;
} dv_endpoint_chip_info_t;

typedef struct dv_endpoint_dram_info {
  uint32_t vendor_id;
  char *vendor_name;
  uint32_t size;
  uint8_t rev_id1;
  uint8_t rev_id2;
  uint8_t density;
  uint8_t io_width;
} dv_endpoint_dram_info_t;

typedef struct dv_endpoint_iface_info {
  dv_endpoint_host_interface_t type;
  int bus_num;
  int device_num;
  union {
    char *pcie_dir;
  } sysfs_path;
  int port_num;
} dv_endpoint_iface_info_t;

typedef struct dv_endpoint_info {
  uint32_t device_id;
  uint32_t vendor_id;
  dv_endpoint_chip_info_t *chip;
  dv_endpoint_dram_info_t *dram;
  dv_endpoint_iface_info_t *iface;
  char *module_name;
  uint32_t gpio0;
  uint32_t gpio1;
  uint32_t device_uid;
} dv_endpoint_info_t;

typedef struct dv_endpoint {
  void *handle;
  dv_session_t *session;
  int num_ep;
  dv_endpoint_group_type_t grp_type;
  dv_endpoint_info_t **ep_info_list;
} dv_endpoint_t;

typedef struct dv_inference_queue_statistics {
  int occupancy_count;
  int length;
  float wait_time;
} dv_inference_queue_statistics_t;

typedef struct dv_model_statistics {
  uint32_t model;
  uint32_t active_input_tensors_count;
  uint32_t active_output_tensors_count;
  uint32_t active_inferences_count;
  uint32_t model_total_dram_occupancy_size;
  uint32_t model_total_input_tensor_occupancy_size;
  uint32_t model_total_output_tensor_occupancy_size;
  dv_model_handle_t *model_handle;
} dv_model_statistics_t;

typedef struct dv_model_input_preprocess_param {
  float qn;
  float *scale;
  float *mean;
  bool aspect_resize;
  bool mirror;
  bool center_crop;
  bool bgr_to_rgb;
  int interpolation;
  bool is_signed;
  int bpp;
  float output_scale;
  float aspect_resize_scale;
  int offset;
  int qmode;
} dv_model_input_preprocess_param_t;

typedef struct dv_model_input_param {
  dv_model_input_preprocess_param_t *preprocess_param;
  int layer_id;
  int blob_id;
  char *layer_name;
  char *blob_name;
  char *layer_type;
  char *layout;
  int size;
  int width;
  int height;
  int depth;
  int nch;
  int bpp;
  int batch_size;
  int num;
  char *src_graph_layer_name;
} dv_model_input_param_t;

typedef struct dv_model_output_postprocess_param {
  float qn;
  bool is_struct_format;
  bool is_float;
  bool is_signed;
  float output_scale;
  int offset;
} dv_model_output_postprocess_param_t;

typedef struct dv_infer_statistics {
  int ep_hw_sys_clk;
  int ep_hw_nnp_clk;
  int ep_hw_sbp_clk;
  int ep_hw_dram_clk;
  uint32_t ep_hw_total_inference_cycles;
  uint32_t ep_hw_fp_cycles;
  float input_transfer_time;
  float output_transfer_time;
  float ep_queue_submission_time;
  uint32_t cumulative_replay_count;
  uint32_t current_replay_count;
  struct timespec input_transfer_start_time_stamp;
  struct timespec output_transfer_start_time_stamp;
  struct timespec inference_start_time_stamp;
  float inference_execution_time;
  uint32_t input_ddr_address;
  uint32_t output_ddr_address;
} dv_infer_statistics_t;

typedef struct dv_infer_llm_info {
  uint32_t llm_infer_resp_num_valid_tokens;
} dv_infer_llm_info_t;

typedef struct dv_infer_request {
  dv_infer_request_handle_t *handle;
  dv_session_t *session;
  dv_endpoint_t *ep_queued;
  dv_endpoint_t *ep_submitted;
  dv_model_t *model;
  dv_blob_t *ip_blob_list;
  dv_blob_t *op_blob_list;
  dv_inference_status_t status;
  dv_infer_statistics_t *stats;
  dv_infer_llm_info_t *llm_infer_info;
} dv_infer_request_t;

const char *dv_stringify_status_code(dv_status_code_t status_code);
dv_status_code_t dv_get_client_lib_version(dv_version_t *client_lib_version);
dv_status_code_t dv_session_create_via_unix_socket(const char *socket_file_path, dv_session_t **session);
dv_status_code_t dv_session_create_via_tcp_ipv4_socket(const char *tcp_ip_addr, int port, dv_session_t **session);
dv_status_code_t dv_session_close(dv_session_t *session);
dv_status_code_t dv_retrieve_version_details(dv_session_t *session, dv_product_version_t **product_version, uint8_t *product_count);
dv_status_code_t dv_exchange_current_version_details(dv_session_t *session, dv_product_version_t **product_version, uint8_t *product_count);
dv_status_code_t dv_free_version_details(dv_product_version_t *product_version);
dv_status_code_t dv_shmfd_register(dv_session_t *session, int fd, uint32_t size, uint32_t offset, int fd_type, dv_shm_descriptor_t **shm_desc);
dv_status_code_t dv_shmfd_unregister(dv_shm_descriptor_t *shm_desc);
dv_status_code_t dv_endpoint_get_list(dv_session_t *session, dv_endpoint_t **ep_list, int *ep_count);
dv_status_code_t dv_endpoint_free_group(dv_endpoint_t *ep_grp);
dv_status_code_t dv_endpoint_check_status(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_state_t *state);
dv_status_code_t dv_endpoint_get_dram_statistics(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_dram_statistics_t **ep_dram_stats, int *ep_count);
dv_status_code_t dv_endpoint_free_dram_statistics(dv_endpoint_dram_statistics_t *ep_dram_stats, int count);
dv_status_code_t dv_endpoint_get_statistics(dv_session_t *session, dv_endpoint_t *ep, dv_endpoint_statistics_t **ep_stats, int *ep_count);
dv_status_code_t dv_endpoint_free_statistics(dv_endpoint_statistics_t *ep_stats, int count);
dv_status_code_t dv_model_load_from_file(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);
dv_status_code_t dv_model_unload(dv_model_t *model);
dv_status_code_t dv_infer_sync(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, int timeout, bool enable_stats, dv_infer_request_t **inf_obj);
dv_status_code_t dv_infer_async(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, bool enable_stats, dv_infer_request_t **inf_obj);
dv_status_code_t dv_infer_wait_for_completion(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **inf_obj);
dv_status_code_t dv_infer_get_req_id(dv_infer_request_t *inf_obj, uint64_t *req_id);
dv_status_code_t dv_infer_free(dv_infer_request_t *inf_obj);
dv_status_code_t dv_infer_get_inflight_count(dv_session_t *session, int *count);

#endif
