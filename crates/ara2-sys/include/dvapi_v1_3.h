/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * DVAPI 1.3 (NXP rt-sdk-ara2 2.1.x): layouts and functions specific to
 * this version. Conventions are described in dvapi_common.h.
 */
#ifndef ARA2_DVAPI_V1_3_H
#define ARA2_DVAPI_V1_3_H

#include "dvapi_common.h"

enum dv_endpoint_state_1_3 {
  DV_ENDPOINT_STATE_THERMAL_ACTIVE_SLOW = 4,
  DV_ENDPOINT_STATE_FAIL_SAFE = 5,
};

typedef enum dv_client_log_level {
  DV_CLIENT_LOG_LEVEL_TRACE = 0,
  DV_CLIENT_LOG_LEVEL_DEBUG = 1,
  DV_CLIENT_LOG_LEVEL_INFO = 2,
  DV_CLIENT_LOG_LEVEL_WARN = 3,
  DV_CLIENT_LOG_LEVEL_ERROR = 4,
  DV_CLIENT_LOG_LEVEL_CRITICAL = 5,
  DV_CLIENT_LOG_LEVEL_OFF = 6,
} dv_client_log_level_t;

typedef enum dv_type_code {
  DV_TYPE_CLIENT_LIB = 0,
  DV_TYPE_PROXY = 1,
  DV_TYPE_DEVICE = 2,
  DV_TYPE_DEVICE_MEMORY = 3,
} dv_component_type_t;

typedef enum dv_criticality_code {
  DV_SEVERITY_OK = 0,
  DV_SEVERITY_RETRYABLE = 1,
  DV_SEVERITY_DEGRADED = 2,
  DV_SEVERITY_FATAL = 3,
} dv_error_severity_t;

typedef uint32_t dv_sub_code_t;

typedef struct dv_status {
  dv_status_code_t main_code;
  dv_sub_code_t sub_code;
  dv_component_type_t type;
  dv_error_severity_t criticality;
} dv_status_t;

struct dv_endpoint_dram_statistics {
  dv_endpoint_t *ep;
  uint64_t ep_total_dram_size;
  uint64_t ep_total_dram_occupancy_size;
  uint64_t ep_total_free_size;
  uint64_t ep_total_reserved_occupancy_size;
  uint64_t ep_total_model_occupancy_size;
  uint64_t ep_total_tensor_occupancy_size;
};

struct dv_endpoint_statistics {
  dv_endpoint_t *ep;
  dv_endpoint_state_t state;
  int ep_sys_clk;
  int ep_dram_clk;
  float ep_core_voltage;
  float ep_temp;
  int num_inference_queues;
  dv_inference_queue_statistics_t *ep_infq_stats;
  int num_active_models;
  dv_model_statistics_t *model_stats;
  struct dv_endpoint_dram_statistics ep_dram_stats;
  dv_endpoint_power_state_t ep_power_state;
  uint32_t ep_soft_reset_count;
  int ep_sbp_clk;
  int ep_nnp_clk;
};

struct dv_model_output_param {
  dv_model_output_postprocess_param_t *postprocess_param;
  int layer_id;
  int blob_id;
  int fused_parent_id;
  char *layer_name;
  char *blob_name;
  char *layer_fused_parent_name;
  char *layer_type;
  char *layout;
  int size;
  int width;
  int height;
  int depth;
  int nch;
  int bpp;
  int num_classes;
  dv_layer_output_type_t layer_output_type;
  int num;
  int max_dynamic_id;
  char *src_graph_layer_name;
  int has_nms_parent;
};

struct dv_model {
  dv_model_handle_t *handle;
  dv_session_t *session;
  dv_endpoint_t *endpoint;
  dv_version_t version;
  char *name;
  dv_model_type_t model_type;
  char *internal_name;
  int num_inputs;
  int num_outputs;
  dv_model_priority_level_t priority;
  dv_model_input_param_t *input_param;
  struct dv_model_output_param *output_param;
  dv_model_llm_params_t *llm_params;
  int num_compiler_config;
  dv_compiler_statistics_t *compiler_stats;
  dv_model_load_options_t *model_load_options;
  bool cp_layer;
};

dv_status_code_t dv_get_endpoint_busyness(dv_session_t *session, dv_endpoint_t *ep, bool *is_busy);
dv_status_code_t dv_client_set_log_level(dv_client_log_level_t log_level);
dv_status_t dv_model_load_from_file_s(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path, const char *model_name, dv_model_priority_level_t priority, dv_model_t **model_handle);
dv_status_t dv_infer_async_s(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array, dv_blob_t *op_array, bool enable_stats, dv_infer_request_t **inf_obj);
dv_status_t dv_infer_wait_for_completion_s(dv_session_t *session, dv_infer_request_t **inf_obj_list, int inf_obj_count, int timeout, dv_infer_request_t **inf_obj);

#endif
