/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
 *
 * DVAPI 1.1 (Kinara ARA-2 runtime 1.2.x): layouts and functions specific
 * to this version. Conventions are described in dvapi_common.h.
 */
#ifndef ARA2_DVAPI_V1_1_H
#define ARA2_DVAPI_V1_1_H

#include "dvapi_common.h"

enum dv_endpoint_state_1_1 {
  DV_ENDPOINT_STATE_ACTIVE_BOOSTED = 4,
  DV_ENDPOINT_STATE_THERMAL_INACTIVE = 5,
};

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

#endif
