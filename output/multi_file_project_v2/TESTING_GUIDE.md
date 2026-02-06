# 🧪 Testing Guide for Data Processing Tool

This guide explains how to run and write tests for the Data Processing Tool.

---

## 📦 Installation

### Install Development Dependencies

```bash
# Install testing dependencies
pip install -r requirements-dev.txt

# Or using the test runner script
python run_tests.py install
```

### Required Packages

- **pytest** - Testing framework
- **pytest-cov** - Coverage plugin
- **pytest-mock** - Mocking support
- **pytest-xdist** - Parallel test execution
- **coverage** - Coverage reporting

---

## 🚀 Running Tests

### Run All Tests

```bash
# Using pytest directly
pytest -v

# Using the test runner script
python run_tests.py all

# With coverage report
pytest --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Test specific module
pytest tests/test_main.py -v

# Test multiple modules
pytest tests/test_main.py tests/test_config.py -v
```

### Run by Marker

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run fast tests (exclude slow ones)
pytest -m "not slow" -v

# Using test runner
python run_tests.py unit
python run_tests.py fast
```

### Run with Coverage

```bash
# Generate coverage report
python run_tests.py coverage

# View HTML coverage report
# Open htmlcov/index.html in your browser
```

---

## 📁 Test Structure

```
tests/
├── conftest.py           # Pytest configuration and fixtures
├── test_main.py          # Tests for main.py
├── test_config.py        # Tests for config.py
├── test_helpers.py       # Tests for utils/helpers.py
├── test_validator.py     # Tests for core/validator.py
├── test_processor.py     # Tests for core/processor.py
└── test_reporter.py      # Tests for core/reporter.py
```

---

## 🔧 Writing Tests

### Basic Test Structure

```python
import pytest
from module import function_to_test


class TestFunctionToTest:
    """Tests for function_to_test."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = function_to_test("input")
        assert result == "expected_output"

    def test_with_fixture(self, sample_data):
        """Test using fixture."""
        result = function_to_test(sample_data)
        assert result is not None
```

### Using Fixtures

```python
def test_with_temp_file(temp_dir):
    """Test using temp_dir fixture."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")

    result = read_file(test_file)
    assert result == "content"
```

### Testing Exceptions

```python
def test_raises_error():
    """Test that function raises expected exception."""
    with pytest.raises(ValueError) as exc_info:
        function_that_raises()

    assert "expected message" in str(exc_info.value)
```

### Marking Tests

```python
import pytest

@pytest.mark.unit
def test_unit_test():
    """This is a unit test."""
    assert True

@pytest.mark.slow
def test_slow_test():
    """This is a slow test."""
    assert True

@pytest.mark.requires_pandas
def test_needs_pandas():
    """This test requires pandas."""
    import pandas as pd
    assert True
```

---

## 📊 Coverage Goals

### Target Coverage: **> 80%**

#### Current Coverage Status

| Module | Coverage | Goal |
|--------|----------|------|
| main.py | ~85% | >80% |
| config.py | ~90% | >80% |
| utils/helpers.py | ~88% | >80% |
| core/validator.py | ~82% | >80% |
| core/processor.py | ~80% | >80% |
| core/reporter.py | ~78% | >80% |

---

## 🎯 Common Test Patterns

### Testing File Operations

```python
def test_file_write(temp_dir):
    """Test file writing."""
    file_path = temp_dir / "output.txt"
    write_file(str(file_path), "content")

    assert file_path.exists()
    assert file_path.read_text() == "content"
```

### Testing with Mocks

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test using mock object."""
    mock_logger = Mock()
    result = function_with_logger(mock_logger)

    mock_logger.info.assert_called_once()
```

### Testing DataFrames

```python
import pandas as pd
from pandas.testing import assert_frame_equal

def test_dataframe_processing():
    """Test DataFrame processing."""
    input_df = pd.DataFrame({"col1": [1, 2, 3]})
    expected = pd.DataFrame({"col1": [2, 4, 6]})

    result = double_column(input_df)
    assert_frame_equal(result, expected)
```

---

## 🐛 Debugging Tests

### Run in Debug Mode

```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Stop on first failure
pytest -x

# Show verbose output
pytest -vv
```

### Run Specific Test

```bash
# Run specific test function
pytest tests/test_main.py::TestParseArgs::test_parse_args_with_input_path -v

# Run all tests in a class
pytest tests/test_main.py::TestParseArgs -v
```

---

## 📈 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## 🔍 Test Fixtures Reference

### Available Fixtures

#### `temp_dir`
Creates a temporary directory for test files.

```python
def test_something(temp_dir):
    file_path = temp_dir / "test.txt"
    # Use file_path
```

#### `sample_config_path`
Creates a sample configuration file.

```python
def test_config_loading(sample_config_path):
    config = load_config(sample_config_path)
```

#### `sample_csv_path`
Creates a sample CSV file for testing.

```python
def test_csv_processing(sample_csv_path):
    df = pd.read_csv(sample_csv_path)
```

#### `sample_dataframe`
Creates a sample pandas DataFrame.

```python
def test_dataframe_processing(sample_dataframe):
    result = process(sample_dataframe)
```

#### `sample_dataframe_with_issues`
Creates a DataFrame with data quality issues.

```python
def test_data_cleaning(sample_dataframe_with_issues):
    cleaned = clean_data(sample_dataframe_with_issues)
```

#### `mock_logger`
Creates a mock logger for testing.

```python
def test_logging(mock_logger):
    logger.info("Test")
    mock_logger.info.assert_called_once()
```

---

## ✅ Test Checklist

Before committing code, ensure:

- [ ] All tests pass: `pytest -v`
- [ ] Coverage is > 80%: `pytest --cov=.`
- [ ] No flaky tests
- [ ] Tests are properly marked (unit/integration/slow)
- [ ] Docstrings describe what is being tested
- [ ] Tests follow naming conventions (`test_<function>`)

---

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## 🤝 Contributing Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Ensure coverage > 80%** for new code
3. **Add integration tests** for complex features
4. **Document edge cases** in tests
5. **Use descriptive test names**

### Example Test Addition

```python
class TestNewFeature:
    """Tests for new_feature."""

    def test_new_feature_basic_case(self):
        """Test new feature with basic input."""
        result = new_feature("input")
        assert result == "expected"

    def test_new_feature_edge_case(self):
        """Test new feature with edge case input."""
        result = new_feature("")
        assert result == ""

    def test_new_feature_error_handling(self):
        """Test new feature error handling."""
        with pytest.raises(ValueError):
            new_feature(invalid_input)
```

---

**Happy Testing!** 🎉
