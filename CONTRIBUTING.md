# Contributing to ARA2

Thank you for your interest in contributing to the ARA2 client library! This
document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Developer Certificate of Origin (DCO)

All contributions must be signed off to certify that you have the right to
submit the code and agree to the [DCO](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### How to Sign Off

Add a `Signed-off-by` line to your commit message:

```
git commit -s -m "Add feature X"
```

This adds a line like:

```
Signed-off-by: Your Name <your.email@example.com>
```

## Getting Started

### Prerequisites

- Rust 2024 edition (see `rust-toolchain.toml`)
- Python 3.11+ (for Python bindings)
- maturin (for building Python bindings)

### Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/EdgeFirstAI/ara2-rs.git
   cd ara2-rs
   ```

2. Build the project:

   ```bash
   cargo build
   ```

3. Run tests:

   ```bash
   cargo test
   ```

4. For Python bindings development:

   ```bash
   cd crates/ara2-py
   maturin develop --release
   ```

## Branch Naming Conventions

Use the following prefixes for branch names:

| Prefix      | Purpose                          | Example                     |
| ----------- | -------------------------------- | --------------------------- |
| `feature/`  | New features                     | `feature/add-batch-api`     |
| `bugfix/`   | Bug fixes                        | `bugfix/fix-memory-leak`    |
| `hotfix/`   | Urgent fixes for production      | `hotfix/security-patch`     |
| `docs/`     | Documentation changes            | `docs/update-readme`        |
| `refactor/` | Code refactoring                 | `refactor/simplify-session` |
| `test/`     | Test additions or modifications  | `test/add-endpoint-tests`   |
| `chore/`    | Maintenance tasks                | `chore/update-deps`         |

Include a JIRA ticket number if applicable:

```
feature/ABC-123-add-batch-api
```

## Commit Message Guidelines

Follow the conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]

Signed-off-by: Your Name <email@example.com>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples

```
feat(session): add TCP/IPv6 socket support

Add support for connecting to the ARA-2 proxy via TCP/IPv6 sockets.
This enables connections in IPv6-only network environments.

Closes #42
Signed-off-by: Developer Name <dev@example.com>
```

```
fix(model): correct tensor buffer alignment

Fix buffer alignment issue that caused crashes on certain architectures
when loading models with non-standard tensor layouts.

Signed-off-by: Developer Name <dev@example.com>
```

## Pull Request Process

1. **Create a branch** following the naming conventions above

2. **Make your changes** with appropriate tests

3. **Ensure quality checks pass:**

   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```

4. **Push and create a Pull Request**

5. **Fill out the PR template** with:
   - Summary of changes
   - Test plan
   - Any breaking changes

6. **Address review feedback**

7. **Ensure CI passes** before requesting merge

### PR Requirements

- [ ] All commits are signed off (DCO)
- [ ] Tests pass (`cargo test`)
- [ ] Linting passes (`cargo clippy`)
- [ ] Formatting is correct (`cargo fmt --check`)
- [ ] Documentation is updated (if applicable)
- [ ] CHANGELOG.md is updated (for user-facing changes)

## Testing

### Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p ara2

# With logging
RUST_LOG=debug cargo test -- --nocapture
```

### Writing Tests

- Place unit tests in a `tests` module within the source file
- Place integration tests in the `tests/` directory
- Use descriptive test names: `test_session_connects_via_unix_socket`

## Code Style

### Rust

- Follow the [Rust Style Guide](https://doc.rust-lang.org/style-guide/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Document public APIs with rustdoc comments

### Python

- Follow PEP 8
- Use type hints
- Document with docstrings (Google style)

## Documentation

- Update `README.md` for user-facing changes
- Update `crates/ara2-py/README.md` for Python-specific changes
- Add rustdoc comments for public APIs
- Update `CHANGELOG.md` for notable changes

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0.

## Questions?

- Open a [GitHub Discussion](https://github.com/EdgeFirstAI/ara2-rs/discussions)
- Email: support@edgefirst.ai
