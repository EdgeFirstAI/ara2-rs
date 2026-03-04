---
name: Bug Report
about: Report a bug to help us improve
title: "[BUG] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Steps to Reproduce

1. ...
2. ...
3. ...

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Environment

- **OS:** (e.g., Ubuntu 22.04, Debian 12)
- **Architecture:** (e.g., x86_64, aarch64)
- **Rust Version:** (output of `rustc --version`)
- **ara2 Version:** (output of `cargo pkgid ara2` or Python `edgefirst_ara2.version()`)

### For Python Issues

- **Python Version:** (output of `python --version`)
- **edgefirst-ara2 Version:** (output of `pip show edgefirst-ara2`)

### For Hardware Issues

- **ARA-2 Firmware Version:**
- **Proxy Version:**

## Code Sample

```rust
// Minimal code to reproduce the issue
```

or

```python
# Minimal code to reproduce the issue
```

## Error Output

```
Paste any error messages or stack traces here
```

## Additional Context

Add any other context about the problem here, such as:
- Model file used (if applicable)
- System logs from `journalctl -u ara2-proxy`
- Whether the issue is reproducible
