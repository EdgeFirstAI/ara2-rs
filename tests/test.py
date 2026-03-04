#!/usr/bin/env python3
"""
Basic tests for the ara2 Python library.
"""

import unittest
import ara2


class TestAra2Library(unittest.TestCase):
    """Test suite for the ara2 Python bindings."""

    def test_version(self):
        """Test that we can get the library version."""
        version = ara2.version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)
        print(f"  Library version: {version}")

    def test_state_enum(self):
        """Test State enum values."""
        self.assertTrue(hasattr(ara2.State, 'Init'))
        self.assertTrue(hasattr(ara2.State, 'Idle'))
        self.assertTrue(hasattr(ara2.State, 'Active'))
        self.assertTrue(hasattr(ara2.State, 'Fault'))

    def test_output_quantization_enum(self):
        """Test OutputQuantization enum values."""
        self.assertTrue(hasattr(ara2.OutputQuantization, 'None'))
        self.assertTrue(hasattr(ara2.OutputQuantization, 'U8'))
        self.assertTrue(hasattr(ara2.OutputQuantization, 'I8'))

    def test_session_creation_error(self):
        """Test that creating a session with invalid path raises an error."""
        with self.assertRaises(Exception) as context:
            ara2.Session.create_via_unix_socket("/invalid/path/to/socket")
        print(f"  Invalid socket path raises error: {context.exception}")

    def test_tcp_session_creation_error(self):
        """Test that creating a TCP session with invalid IP raises an error."""
        with self.assertRaises(Exception) as context:
            ara2.Session.create_via_tcp_ipv4_socket("invalid.ip", 5555)
        print(f"  Invalid IP address raises error: {context.exception}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
