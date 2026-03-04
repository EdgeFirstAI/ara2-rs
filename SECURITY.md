# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue,
please report it responsibly.

### How to Report

**Email:** security@edgefirst.ai

Please include:

1. **Description** - A clear description of the vulnerability
2. **Impact** - The potential impact of the vulnerability
3. **Reproduction steps** - Detailed steps to reproduce the issue
4. **Environment** - OS, Rust version, Python version (if applicable)
5. **Suggested fix** - If you have one (optional)

### What to Expect

- **Acknowledgment:** Within 48 hours of your report
- **Initial assessment:** Within 1 week
- **Resolution target:** Within 90 days for confirmed vulnerabilities

### Disclosure Policy

We follow coordinated disclosure:

1. Reporter submits vulnerability privately
2. We acknowledge and begin assessment
3. We develop and test a fix
4. We release the fix and publish an advisory
5. Reporter may publish details after the fix is released

### Security Advisories

Security advisories are published via:

- GitHub Security Advisories
- Release notes
- Email to registered users (for critical issues)

### Out of Scope

The following are generally out of scope:

- Vulnerabilities in dependencies (report to upstream maintainers)
- Issues requiring physical access to hardware
- Social engineering attacks
- Denial of service attacks

## Security Best Practices

When using this library:

1. **Keep updated** - Use the latest version with security patches
2. **Validate inputs** - Validate model files before loading
3. **Secure connections** - Use appropriate socket permissions
4. **Monitor resources** - Watch for unexpected resource usage

## Contact

For security-related questions that are not vulnerabilities:
support@edgefirst.ai
