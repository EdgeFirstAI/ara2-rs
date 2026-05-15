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
            "Session", "Endpoint", "Model", "InferRequest",
            "InputQuantization", "InputPreprocess", "OutputQuantization",
            "InputTensorInfo", "OutputTensorInfo", "ModelTiming",
            "DvmMetadata", "Ara2Info",
        ]:
            self.assertTrue(hasattr(ara2, name), f"missing {name}")


class TestSessionErrors(unittest.TestCase):
    def test_invalid_unix_socket_raises(self):
        with self.assertRaises(Exception):
            ara2.Session.create_via_unix_socket("/nonexistent/socket")

    def test_invalid_tcp_ip_raises(self):
        with self.assertRaises(Exception):
            ara2.Session.create_via_tcp_ipv4_socket("not.an.ip", 1234)


if __name__ == "__main__":
    unittest.main(verbosity=2)
