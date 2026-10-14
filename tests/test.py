#!/usr/bin/env python3
"""Smoke tests for the edgefirst-ara2 Python bindings.

These tests exercise only no-hardware paths. End-to-end model
execution is covered by the validator cross-validation in Phase 5.
"""

import unittest

import edgefirst_ara2 as ara2


class TestPublicSurface(unittest.TestCase):
    def test_version_string(self):
        self.assertIsInstance(ara2.__version__, str)
        self.assertTrue(ara2.__version__.startswith("0."))

    def test_classes_present(self):
        for name in [
            "Session",
            "Endpoint",
            "Model",
            "InferRequest",
            "InputQuantization",
            "InputPreprocess",
            "OutputQuantization",
            "InputTensorInfo",
            "OutputTensorInfo",
            "ModelTiming",
            "DvmMetadata",
            "Ara2Info",
            "ComponentVersion",
            "InferenceStats",
            "Clocks",
            "EndpointStatistics",
            "EndpointInfo",
            "ChipInfo",
            "DramInfo",
            "Interface",
            "LogLevel",
            "ModelType",
            "UnsupportedError",
            "InterfaceKind",
            "proxy_endpoint",
            "read_model_format",
        ]:
            self.assertTrue(hasattr(ara2, name), f"missing {name}")


class TestSessionErrors(unittest.TestCase):
    def test_invalid_unix_socket_raises(self):
        with self.assertRaises(Exception):
            ara2.Session.create_via_unix_socket("/nonexistent/socket")

    def test_invalid_tcp_ip_raises(self):
        with self.assertRaises(Exception):
            ara2.Session.create_via_tcp_ipv4_socket("not.an.ip", 1234)

    def test_direct_errors_have_status_attributes(self):
        with self.assertRaises(ara2.Ara2Error) as ctx:
            ara2.Session.create_via_tcp_ipv4_socket("not.an.ip", 1234)
        for attr in (
            "code",
            "name",
            "sub_code",
            "component",
            "severity",
            "model_format",
            "compatibility",
        ):
            self.assertIsNone(getattr(ctx.exception, attr), attr)


class TestModelFormat(unittest.TestCase):
    def test_unrecognised_file_has_no_format(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".dvm") as f:
            f.write(b"not a model")
            f.flush()
            self.assertIsNone(ara2.read_model_format(f.name))

    def test_missing_file_raises(self):
        with self.assertRaises(ara2.MetadataError):
            ara2.read_model_format("/nonexistent/model.dvm")

    def test_format_matches_loaded_version(self):
        """Needs a running proxy and ``ARA2_TEST_MODEL``."""
        import os

        path = os.environ.get("ARA2_TEST_MODEL")
        if path is None:
            self.skipTest("ARA2_TEST_MODEL not set")
        fmt = ara2.read_model_format(path)
        model = ara2.Session.connect().list_endpoints()[0].load_model(path)
        self.assertEqual(fmt, int(model.version.split(".")[0]))

    def test_foreign_model_explains_mismatch(self):
        """Needs a running proxy and ``ARA2_TEST_FOREIGN_MODEL``, a model
        built for the other runtime generation."""
        import os

        path = os.environ.get("ARA2_TEST_FOREIGN_MODEL")
        if path is None:
            self.skipTest("ARA2_TEST_FOREIGN_MODEL not set")
        endpoint = ara2.Session.connect().list_endpoints()[0]
        with self.assertRaises(ara2.HardwareError) as ctx:
            endpoint.load_model(path)
        e = ctx.exception
        self.assertEqual(e.code, 520)
        self.assertEqual(e.name, "DV_MODEL_LOAD_FAILURE")
        self.assertEqual(e.compatibility, "mismatch")
        self.assertEqual(e.model_format, ara2.read_model_format(path))
        self.assertIn("reconvert", str(e))


class TestVersionsAndTransport(unittest.TestCase):
    """Need a running proxy; skipped when none is reachable."""

    @classmethod
    def setUpClass(cls):
        try:
            ara2.proxy_endpoint()
        except ara2.ProxyError as e:
            raise unittest.SkipTest(f"no proxy endpoint: {e}")
        cls.session = ara2.Session.connect()

    def test_versions_and_transport(self):
        running = self.session.versions()
        names = [v.component for v in running]
        self.assertIn("proxy", names)
        self.assertIn("client_lib", names)
        self.assertLessEqual(names.count("firmware"), 1, names)
        self.assertGreaterEqual(
            len(self.session.supported_versions()), len(running) - 1
        )
        self.assertIsInstance(self.session.supports_fd_passing, bool)
        self.assertIn(self.session.abi, (ara2.Abi.V1_1, ara2.Abi.V1_3, ara2.Abi.V3_0))

    def test_errors_carry_status_name(self):
        endpoints = self.session.list_endpoints()
        if not endpoints:
            self.skipTest("no endpoints")
        endpoint = endpoints[0]
        with self.assertRaises(ara2.Ara2Error) as ctx:
            endpoint.load_model("/nonexistent/model.dvm")
        self.assertIsNotNone(ctx.exception.name)
        self.assertTrue(ctx.exception.name.startswith("DV_"), ctx.exception.name)
        self.assertIsInstance(ctx.exception.code, int)


class TestTelemetry(unittest.TestCase):
    """Need a running proxy and ``ARA2_TEST_MODEL``."""

    def test_telemetry(self):
        import os
        import time

        if "ARA2_TEST_MODEL" not in os.environ:
            self.skipTest("ARA2_TEST_MODEL not set")
        session = ara2.Session.connect()
        endpoint = session.list_endpoints()[0]
        info = endpoint.info()
        self.assertEqual(info.vendor_id, 0x1E58)
        self.assertIsNotNone(info.chip)
        self.assertGreater(info.chip.neural_processors, 0)
        model = endpoint.load_model(os.environ["ARA2_TEST_MODEL"])
        model.allocate_tensors(None)
        before = time.monotonic()
        stats = model.run()
        after = time.monotonic()
        self.assertTrue(stats.timing.run_time > 0 and stats.hw_cycles > 0)
        # None on DVAPI 1.x, whose runtime stamps with the wall clock.
        if stats.inference_start is not None:
            self.assertTrue(before <= stats.inference_start <= after)
        es = endpoint.statistics()
        self.assertTrue(es.clocks.sys_mhz > 0 and es.dram.dram_size > 0)
        self.assertIsInstance(es.soft_reset_count, int)
        if session.abi == ara2.Abi.V1_1:
            try:
                self.assertIsInstance(endpoint.is_busy(), bool)
            except ara2.UnsupportedError:
                pass
        else:
            self.assertIsInstance(endpoint.is_busy(), bool)
        self.assertEqual(model.model_type, ara2.ModelType.Ara2Cnn)

        self.assertIsInstance(model.version, str)
        self.assertIsInstance(model.name, str)
        if session.abi == ara2.Abi.V3_0:
            self.assertIsNotNone(stats.inference_start)
            largest = es.dram.largest_contiguous_free
            self.assertTrue(0 < largest <= es.dram.free_size)
        else:
            self.assertIsNone(stats.input_start)
            self.assertIsNone(stats.inference_start)
            self.assertIsNone(stats.output_start)
            self.assertIsNone(es.dram.largest_contiguous_free)
        if session.abi == ara2.Abi.V1_1:
            with self.assertRaises(ara2.UnsupportedError):
                session.set_log_level(ara2.LogLevel.Warn)
        else:
            session.set_log_level(ara2.LogLevel.Warn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
