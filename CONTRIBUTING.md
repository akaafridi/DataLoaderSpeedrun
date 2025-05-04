# Contributing to grok3-io-opt

Thank you for your interest in contributing to grok3-io-opt! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, PyTorch version)
6. Any relevant logs or error messages

### Suggesting Enhancements

If you have ideas for enhancements, please create an issue with:

1. A clear, descriptive title
2. A detailed description of the enhancement
3. Any relevant use cases or examples
4. Why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request

## Development Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/grok3-io-opt.git
cd grok3-io-opt

# Install development dependencies
pip install -e ".[dev]"

# Build the C++ extension
python setup.py build_ext --inplace
```

### Using Docker

```bash
# Build and run the Docker container
docker-compose build
docker-compose run benchmark
```

## Testing

Run the tests with:

```bash
pytest
```

## Coding Standards

- Follow PEP 8 guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate

## Benchmarking

Before submitting a PR, please run the benchmarks to ensure your changes don't cause performance regressions:

```bash
python -m grok3_io_opt.benchmark --multi-seed
```

## Documentation

Update documentation for any changes to the API or behavior:

1. Update docstrings
2. Update README.md with any new features or changes
3. Add examples if necessary

## Releasing

For maintainers only:

1. Update version in `__init__.py` and `setup.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Build and upload the package to PyPI

```bash
python -m build
twine upload dist/*
```

Thank you for contributing to grok3-io-opt!