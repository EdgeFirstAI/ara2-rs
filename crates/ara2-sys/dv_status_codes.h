/*
 * Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
 * Kinara  Proprietary. This software is owned or controlled by Kinara and may only be used strictly in accordance with the applicable license terms.
 */

#ifndef __DV_ERR_DEFINES_H__
#define __DV_ERR_DEFINES_H__

typedef enum dv_error_category {
  DV_ERROR_CATEGORY_SUCCESS = 0,
  DV_ERROR_CATEGORY_RETRY = 100,
  DV_ERROR_CATEGORY_INVALID = 200,
  DV_ERROR_CATEGORY_SW_CLIENT_FATAL = 300,
  DV_ERROR_CATEGORY_SW_SERVER_FATAL = 400,
  DV_ERROR_CATEGORY_HW_FATAL = 500,
} dv_error_category_t;

typedef enum dv_status_code {
  // DV_ERROR_CATEGORY_SUCCESS
  DV_ERROR_CATEGORY_SUCCESS_START = 0,
  DV_SUCCESS = 0,
  DV_ERROR_CATEGORY_SUCCESS_END = 0,
  // TODO KARTHIK only change name to DV_STATUS_UNKNOWN
  DV_FAILURE_UNKOWN = 1,

  // DV_ERROR_CATEGORY_RETRY
  DV_ERROR_CATEGORY_RETRY_START = 100,
  DV_ENDPOINT_OUT_OF_MEMORY = 100,  // Not enough DDR memory available on endpoint; use
                                    // dv_endpoint_get_dram_statistics to get available memory
                                    // inference queue (Internal to Kinara)
  DV_ERROR_CATEGORY_RETRY_END = 199,

  // DV_ERROR_CATEGORY_INVALID
  DV_ERROR_CATEGORY_INVALID_START = 200,
  DV_RESOURCE_NOT_FOUND = 200,  // Model file given as input not accessible
  DV_INVALID_VALUE = 201,       // Invalid pointer value encountered in proxy (Internal to Kinara)
  DV_INVALID_HOST_PTR = 202,    // Invalid pointer value encountered in clientlib
  DV_INVALID_OPERATION = 203,   // One of the below invalid requests sent
                                //   1. SHM/SFD request over TCP/IP socket
  //   2. Snapshot dump recieved invalid endpoint ID
  //   3. Proxy received invalid request ID (Internal to Kinara)

  DV_OPERATION_NOT_PERMITTED = 204,            // Reserved (unused currently)
  DV_OPERATION_NOT_SUPPORTED = 205,            // Reserved (unused currently)
  DV_SESSION_UNIX_SOCKET_FILE_TOO_LONG = 220,  // Unix socket file exceeded 108 bytes
  DV_SESSION_INVALID_TCP_IPV4_ADDR = 221,      // Invalid TCP Address
  DV_SESSION_INVALID_TCP_IPV4_PORT = 222,      // Invalid TCP port value
  DV_SESSION_INVALID_HANDLE = 223,             // Request sent over session that does not exist
  DV_ENDPOINT_INVALID_HANDLE = 230,            // Request sent for endpoint list with no active endpoints
  DV_ENDPOINT_INVALID_PARAMS = 231,            // Invalid endpoint param sent for stats collection/fault recovery
  DV_ENDPOINT_NOT_FOUND = 232,                 // Reserved (currently unused)
  DV_ENDPOINT_NOT_AVAILABLE = 233,             // No valid endpoints received in dv_endpoint_get_list
  DV_ENDPOINT_GROUP_INVALID = 234,             // Reserved (unused currently)
  DV_ENDPOINT_POWER_SWITCH_FAILURE = 235,      // Endpoint power switch failed
  DV_ENDPOINT_POWER_GATED = 236,               // Endpoint cannot run inferences since it is in power gated mode
  DV_ENDPOINT_INVALID_POWER_STATE = 237,       // Invalid power state switch requested
  DV_ENDPOINT_GET_LIST_FAILED = 238,           // Generic error if get endpoint list fails.
  DV_ENDPOINT_GET_STATS_FAILED = 239,
  DV_MODEL_INVALID_PARAMS = 240,  // One of the below errors occurred
  //  1. Model context not found/ invalid on device (Internal to
  //  Kinara)
  //  2. Inference request sent on a device with no model loaded
  //  3. Invalid input size sent for model
  DV_MODEL_INVALID_HANDLE = 241,  // One of the below errors occurred
  //  1. Null/invalid model handle passed for inference
  //  2. Invalid model handle passed to dv_model_unload
  DV_MODEL_INVALID_MODEL_FILE = 242,   // Model of size 0 passed
  DV_MODEL_PARSE_FAILURE = 243,        // Invalid file format of model.dvm
  DV_MODEL_UNSUPPORTED_VERSION = 244,  // Deprecated model.dvm file provided
  DV_MODEL_CACHING_FAILURE = 245,      // One of the below errors occurred
  //  1. Model cache path inaccessible for model write
  //  2. Model cache path not writable
  DV_MODEL_CACHE_FETCH_FAILURE = 246,  // One of the below errors occurred
  //  1. Model cache path inaccessible for model read
  //  2. Model not readable from cache
  //  3. Checksum comparison failed after model read from cache

  DV_MODEL_SET_LLM_CFG_PARAMS_FAILED = 247,  // unable to set llm cfg params
  DV_SOFT_RESET_FAILURE = 248,               // unable to do soft reset
  DV_SOFT_RESET_UNSUPPORTED = 249,           // Soft reset unsupported

  //  (Internal to Kinara)
  DV_INFER_INVALID_SHM_BUFFERS = 250,     // Unable to read/write SHM buffers
  DV_INFER_REQUEST_INVALID_HANDLE = 251,  // dv_wait_for_completion called on invalid inference request object
                                          // (or) error in clientlib while processing inference
  DV_FIRMWARE_VERSION_INVALID = 252,      // invalid firmware version for firmware binaries
  DV_MODEL_LOAD_SUBMITTED = 253,
  DV_MODEL_ALREADY_LOADED = 254,
  DV_MODEL_UNLOAD_SUBMITTED = 255,
  DV_MODEL_LOAD_SUBMIT_FAILED = 256,
  DV_MODEL_LOAD_ABORTED = 257,
  DV_MODEL_UNLOAD_SUBMIT_FAILED = 258,
  DV_ERROR_CATEGORY_INVALID_END = 299,

  // DV_ERROR_CATEGORY_SW_CLIENT_FATAL
  DV_ERROR_CATEGORY_SW_CLIENT_FATAL_START = 300,
  DV_CLIENT_VERSION_MISMATCH = 300,  // Incorrect version of clientlib used; to
                                     // be matched with Proxy version
  DV_CONNECTION_ERROR = 301,         // Connection is faulty/closed

  DV_HIF_PUSH_FAILED = 302,
  DV_HIF_ERROR = 303,
  DV_HIF_TIMEOUT = 304,
  DV_HIF_POP_FAILED = 305,
  DV_HIF_DEVICE_IN_CONFIG = 306,
  DV_ERROR_CATEGORY_SW_CLIENT_FATAL_END = 399,

  // DV_ERROR_CATEGORY_SW_SERVER_FATAL
  DV_ERROR_CATEGORY_SW_SERVER_FATAL_START = 400,
  DV_HOST_OUT_OF_MEMORY = 400,         // Host is out of memory (calloc/malloc call fails)
  DV_INTERNAL_ERROR = 401,             // One of the below errors occurred (Internal to Kinara)
                                       // 1. Mismatch in expected payload sizes
                                       // 2. Failed to add request to host list
  DV_REQUEST_TIMEDOUT = 402,           // Request timed out; default values of timeouts
                                       // are API specific and mentioned in dvapi.h
  DV_SHMBUF_NOT_PERMITTED = 403,       // One of the below errors occurred
                                       // 1. Mmap of shm file failed
                                       // 2. Invalid fd sent in sfd request
                                       // 3. No more free shm buffers present
  DV_REQUEST_SEND_FAILED = 404,        // failed to send the request
  DV_REQUEST_PROCESSING_FAILED = 405,  // failed to processing incoming request
                                       // 1. happens at on_request()

  DV_ERROR_CATEGORY_SW_SERVER_FATAL_END = 499,

  // DV_ERROR_CATEGORY_HW_FATAL (Endpoint to be reset)
  DV_ERROR_CATEGORY_HW_FATAL_START = 500,
  DV_ENDPOINT_DMA_FAILED = 500,             // DMA from endpoint to host failed in case of PCIe
  DV_ENDPOINT_FIRMWARE_LOAD_FAILURE = 501,  // Failed to load firmware
  DV_ENDPOINT_FIRMWARE_BOOT_FAILURE = 502,  // Failed to boot firmware
  DV_ENDPOINT_NO_FIRMWARE = 503,            // Reserved (unused currently)
  DV_ENDPOINT_NOT_REACHABLE = 504,          // Interface gone bad or device exception occurred (Refer Fault
                                            // Handling document for further details)
  DV_ENDPOINT_MODEL_BINDING_FAILURE = 505,  // Model binding not present in the  device.
  DV_TENSOR_FREE_ERROR = 506,               // Failed to free the allocated tensors

  DV_MODEL_LOAD_FAILURE = 520,    // Model load request failed on all endpoints in list due to one of
                                  // the following
                                  //  1. Model write to endpoints failed
                                  //  2. Model context write to endpoints failed
                                  //  3. Unable to perform model integrity check after model write
                                  //  (Internal to Kinara)
                                  //  4. Model checksum compare failure (Internal to Kinara)
                                  //  5. Invalid model object received in request
                                  //  6. No endpoints active in endpoint list to load model
  DV_MODEL_RELOAD_FAILURE = 521,  // Device in faulty state on trying to reload model from cache
  DV_MODEL_UNLOAD_FAILURE = 522,
  DV_PARTIAL_SUCCESS = 523,                 // failed to load/unload model on some of the given devices
  DV_TENSOR_WRITE_FAILURE = 541,            // Failed to write buffer to endpoint (Internal to Kinara)
  DV_TENSOR_READ_FAILURE = 542,             // Failed to read buffer from endpoint  (Internal to Kinara)
  DV_TENSOR_CREATE_FAILURE = 543,           // Failed to create tensor
  DV_TENSOR_ALLOCTAION_FAILURE = 545,       // Failed to allocate device memory for a tensor.
  DV_TENSOR_INTEGRITY_CHECK_FAILURE = 546,  // Integrity check failed.

  DV_PROXY_DEVICE_INIT_FAILURE = 551,  // Proxy not able to initialize device

  DV_INFER_TIME_OUT = 560,                  // Inference request failed to complete within the timeout limit
  DV_INFER_FAILURE = 561,                   // inference failure
  DV_INFER_INVALID_INPUT = 562,             // inference request received with invalid input
  DV_INFER_QUEUE_FULL = 563,                // Inference queue on host side is full; current inferences in queue
                                            // to be processed to accept more requests
  DV_INFER_QUEUE_EMPTY = 564,               // No inferences scheduled on endpoint host
  DV_INFER_MODEL_NOT_FOUND = 565,           // Model not loaded on for inference request
  DV_INFER_ABORTED = 566,                   // Inference aborted before submitting to device
  DV_INFER_SUBMIT_FAILURE = 567,            // Infer request submission to device failed
  DV_INFER_TIME_OUT_THERMAL_RUNAWAY = 568,  // Inference request failed due to increase in temperature

  DV_ERROR_CATEGORY_HW_FATAL_END = 599,

  DV_CLIENT_TXRX_WRITE_FAILURE = 600,         // UV write failue
  DV_CLIENT_TXRX_READ_FAILURE = 601,          // UV read failure
  DV_CLIENT_TXRX_ASYNC_SEND_FAILURE = 602,    // UV read failure
  DV_CLIENT_TXRX_FD_COUNT_MISMATCH = 603,     // fd count mismatch at proxy client
  DV_CLIENT_TXRX_DISCONNECT_ERROR = 604,      // client disconnected from server
  DV_CLIENT_RECEIVED_UNKNOWN_RESPONSE = 605,  // proxy client received unknown response
  DV_CLIENT_TO_TXRX_CONNECTION_ERROR = 606,   // client not connnected to transreceiver

  DV_FLOW_CREATE_FAILED = 700,  // Failed to submit the control flow
  DV_FLOW_SUBMIT_FAILED = 701,  // Failed to submit the control flow
  DV_FLOW_ABORTED = 702,        // Flow is aborted explicitly

  DV_CP_INDEX_ALLOC_FAILURE = 800,  // Failed to get the CP index.
  DV_MMIO_READ_FAIL = 900,
  DV_MMIO_WRITE_FAIL = 901,
} dv_status_code_t;

#endif  // __DV_ERR_DEFINES_H__
