"""
Unit tests for core/validator.py module.

Tests data validation logic, schema validation, and type checking.
"""

import pytest

from core.validator import (
    SchemaError,
    ValidationError,
    check_types,
    validate_schema,
)


class TestCheckTypes:
    """Tests for check_types function."""

    def test_check_types_int_matches(self):
        """Test type check with matching integer."""
        result = check_types(10, int)

        assert result is True

    def test_check_types_str_matches(self):
        """Test type check with matching string."""
        result = check_types("hello", str)

        assert result is True

    def test_check_types_list_matches(self):
        """Test type check with matching list."""
        result = check_types([1, 2, 3], list)

        assert result is True

    def test_check_types_tuple_of_types(self):
        """Test type check with tuple of allowed types."""
        result = check_types(10, (int, float))

        assert result is True

    def test_check_types_float_in_tuple(self):
        """Test type check with float in tuple."""
        result = check_types(10.5, (int, float))

        assert result is True

    def test_check_types_mismatch(self):
        """Test type check with mismatched type."""
        result = check_types([], int)

        assert result is False

    def test_check_types_string_not_int(self):
        """Test that string doesn't match int type."""
        result = check_types("hello", int)

        assert result is False

    def test_check_types_invalid_expected_type(self):
        """Test that invalid expected_type raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            check_types(10, "int")

        assert "must be a type" in str(exc_info.value).lower()

    def test_check_types_with_none(self):
        """Test type check with None value."""
        result = check_types(None, type(None))

        assert result is True


class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_schema_success(self):
        """Test successful schema validation."""
        data = {
            "name": "Alice",
            "age": 25,
            "active": True
        }
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True},
            "active": {"type": bool, "required": False}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_validate_schema_missing_required_key(self):
        """Test validation fails with missing required key."""
        data = {
            "name": "Alice"
            # Missing "age" which is required
        }
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True}
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_schema(data, schema)

        assert "missing" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()

    def test_validate_schema_wrong_type(self):
        """Test validation fails with wrong data type."""
        data = {
            "name": "Alice",
            "age": "25"  # Should be int
        }
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True}
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_schema(data, schema)

        assert "age" in str(exc_info.value).lower()

    def test_validate_schema_with_default_value(self):
        """Test validation uses default value for missing optional key."""
        data = {
            "name": "Alice"
        }
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": False, "default": 18}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_validate_schema_allows_extra_keys(self):
        """Test that validation allows extra keys not in schema."""
        data = {
            "name": "Alice",
            "age": 25,
            "extra_field": "some value"
        }
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_validate_schema_tuple_of_types(self):
        """Test validation with tuple of allowed types."""
        data = {
            "value": 42
        }
        schema = {
            "value": {"type": (int, float), "required": True}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_validate_schema_empty_data(self):
        """Test validation with empty data dict."""
        data = {}
        schema = {
            "name": {"type": str, "required": False}
        }

        result = validate_schema(data, schema)

        assert result is True


class TestValidatorExceptions:
    """Tests for custom exception classes."""

    def test_validation_error_can_be_raised(self):
        """Test that ValidationError can be raised."""
        with pytest.raises(ValidationError):
            raise ValidationError("Validation failed")

    def test_validation_error_is_exception(self):
        """Test that ValidationError inherits from Exception."""
        with pytest.raises(Exception):
            raise ValidationError("Test")

    def test_schema_error_can_be_raised(self):
        """Test that SchemaError can be raised."""
        with pytest.raises(SchemaError):
            raise SchemaError("Invalid schema")

    def test_schema_error_is_exception(self):
        """Test that SchemaError inherits from Exception."""
        with pytest.raises(Exception):
            raise SchemaError("Test")


class TestComplexValidationScenarios:
    """Tests for complex validation scenarios."""

    def test_nested_dict_validation(self):
        """Test validation with nested dictionaries."""
        data = {
            "user": {
                "name": "Alice",
                "age": 25
            }
        }
        schema = {
            "user": {"type": dict, "required": True}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_list_validation(self):
        """Test validation with list values."""
        data = {
            "tags": ["python", "data", "processing"]
        }
        schema = {
            "tags": {"type": list, "required": True}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_multiple_required_fields(self):
        """Test validation with multiple required fields."""
        data = {
            "field1": "value1",
            "field2": "value2",
            "field3": "value3"
        }
        schema = {
            "field1": {"type": str, "required": True},
            "field2": {"type": str, "required": True},
            "field3": {"type": str, "required": True}
        }

        result = validate_schema(data, schema)

        assert result is True

    def test_validation_with_none_values(self):
        """Test validation handles None values correctly."""
        data = {
            "field1": None,
            "field2": "value"
        }
        schema = {
            "field1": {"type": type(None), "required": False},
            "field2": {"type": str, "required": True}
        }

        result = validate_schema(data, schema)

        assert result is True
