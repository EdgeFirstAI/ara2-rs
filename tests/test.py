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
        ]:
            self.assertTrue(hasattr(ara2, name), f"missing {name}")


class TestSessionErrors(unittest.TestCase):
    def test_invalid_unix_socket_raises(self):
        with self.assertRaises(Exception):
            ara2.Session.create_via_unix_socket("/nonexistent/socket")

    def test_invalid_tcp_ip_raises(self):
        with self.assertRaises(Exception):
            ara2.Session.create_via_tcp_ipv4_socket("not.an.ip", 1234)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
