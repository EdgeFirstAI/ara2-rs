/*
 * DVAPI 1.3 additions over the 1.1 base in dvapi.h.
 *
 * Declarations here were derived from Kinara's DVAPI 1.3.2 dvapi.h and
 * dv_status_codes.h. Only what 1.3 adds appears; everything else is
 * unchanged from 1.1 and comes from the base header.
 *
 * Structs 1.3 extends are declared in full under a _1_3 suffix rather than
 * replacing the base definition. Every 1.3 change is a tail append, so the
 * base layout stays valid for reads under either version and the suffixed
 * type supplies the size when a stride is needed.
 *
 * Enums cannot be extended in C, so 1.3's additions to dv_endpoint_state_t
 * and dv_status_code_t are declared as separate enums. For the endpoint
 * state that is not cosmetic: 1.3 reuses values 4 and 5 for different
 * names, so the two enums deliberately disagree at those values and the
 * loaded library's DVAPI version selects which set applies.
 */

#ifndef __DV_API_1_3_H__
#define __DV_API_1_3_H__

#include "dvapi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- New types ------------------------------------------------------- */

typedef enum dv_client_log_level {
  DV_CLIENT_LOG_LEVEL_TRACE = 0,    /**< dump trace logs */
  DV_CLIENT_LOG_LEVEL_DEBUG = 1,    /**< dump debug logs */
  DV_CLIENT_LOG_LEVEL_INFO = 2,     /**< dump info logs */
  DV_CLIENT_LOG_LEVEL_WARN = 3,     /**< dump warning logs */
  DV_CLIENT_LOG_LEVEL_ERROR = 4,    /**< dump error logs */
  DV_CLIENT_LOG_LEVEL_CRITICAL = 5, /**< dump critical logs */
  DV_CLIENT_LOG_LEVEL_OFF = 6,      /**< turn off client logs */
} dv_client_log_level_t;

typedef struct dv_stats_options {
  uint32_t state_type;
  uint32_t temp_threshold;
} dv_stats_options_t;

typedef enum dv_type_code {
  DV_TYPE_CLIENT_LIB = 0,
  DV_TYPE_PROXY = 1,
  DV_TYPE_DEVICE = 2,
  DV_TYPE_DEVICE_MEMORY = 3,
} dv_component_type_t;

typedef enum dv_criticality_code {
  DV_SEVERITY_OK = 0,        /**< fully functional */
  DV_SEVERITY_RETRYABLE = 1, /**< temporary issue, retry possible */
  DV_SEVERITY_DEGRADED = 2,  /**< functional but not as expected */
  DV_SEVERITY_FATAL = 3,     /**< non-recoverable, abort/escalate */
} dv_error_severity_t;

typedef uint32_t dv_sub_code_t;

typedef struct dv_status {
  dv_status_code_t main_code;      /**< primary error code (backward compatible) */
  dv_sub_code_t sub_code;          /**< secondary error details */
  dv_component_type_t type;        /**< error category */
  dv_error_severity_t criticality; /**< severity level */
} dv_status_t;

/* --- Endpoint state ---------------------------------------------------
 *
 * 1.3 renamed values 4 and 5. The doc comments were left as they were in
 * 1.1, and for value 4 they contradict the 1.1 name while agreeing with
 * the 1.3 one, so 4 most likely always meant "reduced frequency". Value 5
 * is undetermined: its comment still backs the 1.1 name. Both sets are
 * kept so the name reported matches the header of the library actually
 * loaded.
 */

typedef enum DV_ENDPOINT_STATE_1_3 {
  DV_ENDPOINT_STATE_1_3_INIT = 0,                /**< endpoint is in init state */
  DV_ENDPOINT_STATE_1_3_IDLE = 1,                /**< endpoint is in idle state */
  DV_ENDPOINT_STATE_1_3_ACTIVE = 2,              /**< endpoint is in active state */
  DV_ENDPOINT_STATE_1_3_ACTIVE_SLOW = 3,         /**< endpoint is operating at reduced frequency */
  DV_ENDPOINT_STATE_1_3_THERMAL_ACTIVE_SLOW = 4, /**< endpoint is operating at reduced frequency */
  DV_ENDPOINT_STATE_1_3_FAIL_SAFE = 5,           /**< endpoint is in thermal Inactive state */
  DV_ENDPOINT_STATE_1_3_THERMAL_UNKNOWN = 6,     /**< endpoint is in unknown thermal state */
  DV_ENDPOINT_STATE_1_3_INACTIVE = 7,            /**< endpoint is in Inactive state */
  DV_ENDPOINT_STATE_1_3_FAULT = 8,               /**< endpoint is in faulty state */
  DV_ENDPOINT_STATE_1_3_BAD_INTERFACE = 1001,    /**< [unsupported] */
  DV_ENDPOINT_STATE_1_3_RECOVERY = 1003,         /**< [unsupported] */
  DV_ENDPOINT_STATE_1_3_DEAD = 1004,             /**< [unsupported] */
  DV_ENDPOINT_STATE_1_3_DRAIN = 1005,            /**< [unsupported] */
  DV_ENDPOINT_STATE_1_3_POWER_GATED = 1006,      /**< [unsupported] */
  DV_ENDPOINT_STATE_1_3_CLOSED = 1007,           /**< [unsupported] */
} dv_endpoint_state_1_3_t;

/* --- Status codes added in 1.3 ----------------------------------------
 *
 * 506 is reused: DV_TENSOR_FREE_ERROR in 1.1, DV_ENDPOINT_DYN_POWER_SET_FAILURE
 * in 1.3, with the tensor error moving to 519.
 */

typedef enum dv_status_code_1_3 {
  DV_1_3_ERROR_NULLPTR = 310,                   /**< null pointer passed to library */
  DV_1_3_ERROR_DEST_TOO_SMALL = 311,            /**< destination buffer is small */
  DV_1_3_ERROR_SRC_TOO_SMALL = 312,             /**< source buffer is small */
  DV_1_3_ERROR_MEMCPY_FAILED = 313,             /**< memcpy_s failed */
  DV_1_3_ERROR_MEMSET_FAILED = 314,             /**< memset_s failed */
  DV_1_3_ERROR_STRNCPY_FAILED = 315,            /**< strncpy_s failed */
  DV_1_3_ENDPOINT_DYN_POWER_SET_FAILURE = 506,  /**< unable to set dynamic power switch idle time */
  DV_1_3_TENSOR_FREE_ERROR = 519,               /**< failed to free the allocated tensors */
  DV_1_3_INFER_TOKEN_OVERFLOW = 569,            /**< inference request failed due to token overflow */
} dv_status_code_1_3_t;

/* --- Extended structs -------------------------------------------------
 *
 * Tail appends over their 1.1 counterparts. The base type remains correct
 * for reading fields both versions share; these supply the 1.3 size.
 */

typedef struct dv_model_output_param_1_3 {
  dv_model_output_postprocess_param_t *postprocess_param;
  int layer_id;                             /**< layer id */
  int blob_id;                              /**< input blob id within the layer */
  int fused_parent_id;                      /**< layer fused parent id */
  char *layer_name;                         /**< layer name */
  char *blob_name;                          /**< input blob name within the layer */
  char *layer_fused_parent_name;            /**< layer fused parent name */
  char *layer_type;                         /**< layer type */
  char *layout;                             /**< output layout */
  int size;                                 /**< layer size in bytes */
  int width;                                /**< layer width in pixels */
  int height;                               /**< layer height in pixels */
  int depth;                                /**< layer depth in pixels */
  int nch;                                  /**< number of channels */
  int bpp;                                  /**< bytes per pixel */
  int num_classes;                          /**< number of classes the model is trained on */
  dv_layer_output_type_t layer_output_type; /**< output type of layer */
  int num;                                  /**< num dimension */
  int max_dynamic_id;                       /**< max batch id */
  char *src_graph_layer_name;               /**< source graph output layer name */
  int has_nms_parent;                       /**< 1, if any parent layer is NMS -- added in 1.3 */
} dv_model_output_param_1_3_t;

/* 1.3 appended two clock fields. The typedef name dv_endpoint_statistics_t
 * is unchanged from 1.1 -- only the struct tag differs upstream -- so the
 * extended form is declared under a suffixed name. */
typedef struct dv_endpoint_statistics_1_3 {
  dv_endpoint_t *ep;                              /**< endpoint handle */
  dv_endpoint_state_t state;                      /**< endpoint state */
  int ep_sys_clk;                                 /**< endpoint system core clock in MHz */
  int ep_dram_clk;                                /**< endpoint dram clock in MHz */
  float ep_core_voltage;                          /**< average endpoint core voltage in volts */
  float ep_temp;                                  /**< average endpoint temperature in degrees celsius */
  int num_inference_queues;                       /**< [unsupported] */
  dv_inference_queue_statistics_t *ep_infq_stats; /**< [unsupported] */
  int num_active_models;                          /**< [unsupported] */
  dv_model_statistics_t *model_stats;             /**< [unsupported] */
  dv_endpoint_dram_statistics_t ep_dram_stats;    /**< endpoint dram statistics */
  dv_endpoint_power_state_t ep_power_state;       /**< [unsupported] */
  uint32_t ep_soft_reset_count;                   /**< non zero for usb devices */
  int ep_sbp_clk;                                 /**< endpoint sbp clock in MHz -- added in 1.3 */
  int ep_nnp_clk;                                 /**< endpoint nnp clock in MHz -- added in 1.3 */
} dv_endpoint_statistics_1_3_t;

/* 1.3 appended one flag. As above, dv_model_load_options_t is already the
 * 1.1 typedef name; only the struct tag differs upstream. */
typedef struct dv_model_load_options_1_3 {
  char *model_name;                   /**< model name */
  dv_model_priority_level_t priority; /**< priority of the model [unused] */
  bool cache;                         /**< if true, the model is cached on disk */
  bool async;                         /**< if true, the load API returns immediately */
  dv_model_type_t model_type;         /**< model type, defaults to DV_MODEL_TYPE_ARA2_CNN */
  bool send_model_filepath_to_proxy;  /**< send the path instead of the blob -- added in 1.3 */
} dv_model_load_options_1_3_t;

/* --- Functions added in 1.3 ------------------------------------------- */

EXPORT
dv_status_code_t dv_client_set_log_level(dv_client_log_level_t log_level);

EXPORT
dv_status_code_t dv_endpoint_get_statistics_with_options(dv_session_t *session, dv_endpoint_t *ep,
                                                         dv_endpoint_statistics_1_3_t **ep_stats, int *ep_count,
                                                         dv_stats_options_t *stats_options);

EXPORT
dv_status_t dv_model_load_from_file_s(dv_session_t *session, dv_endpoint_t *endpt, const char *model_file_path,
                                      const char *model_name, dv_model_priority_level_t priority,
                                      dv_model_t **model_handle);

EXPORT
dv_status_t dv_model_load_from_file_with_options_s(dv_session_t *session, dv_endpoint_t *endpt,
                                                   const char *model_file_path, dv_model_t **model_handle,
                                                   dv_model_load_options_1_3_t *options);

EXPORT
dv_status_t dv_model_load_from_blob_with_options_s(dv_session_t *session, dv_endpoint_t *endpt, dv_blob_t *blob,
                                                   dv_model_t **model_handle, dv_model_load_options_1_3_t *options);

EXPORT
dv_status_t dv_infer_async_s(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model, dv_blob_t *ip_array,
                             dv_blob_t *op_array, bool enable_stats, dv_infer_request_t **inf_obj);

EXPORT
dv_status_t dv_infer_async_with_options_s(dv_session_t *session, dv_endpoint_t *endpt, dv_model_t *model,
                                          dv_blob_t *ip_array, dv_blob_t *op_array, dv_infer_request_t **inf_obj,
                                          dv_infer_options_t *infer_options);

EXPORT
dv_status_t dv_infer_wait_for_completion_s(dv_session_t *session, dv_infer_request_t **inf_obj_list,
                                           int inf_obj_count, int timeout, dv_infer_request_t **inf_obj);

#ifdef __cplusplus
}
#endif

#endif  // __DV_API_1_3_H__
