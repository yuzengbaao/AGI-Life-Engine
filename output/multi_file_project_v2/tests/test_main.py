"""
Unit tests for main.py module.

Tests command-line argument parsing, logging setup, and main execution flow.
"""

import logging
import sys
from unittest.mock import Mock, patch

import pytest

from main import parse_args, setup_logging, run_app, main


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_with_input_path(self):
        """Test parsing with just input path."""
        args = parse_args(["data.csv"])
        assert args.input_path == "data.csv"
        assert args.output_file is None
        assert args.verbose is False
        assert args.config is None

    def test_parse_args_with_all_options(self):
        """Test parsing with all options."""
        args = parse_args([
            "data.csv",
            "-o", "output.xlsx",
            "--verbose",
            "--config", "config.yaml"
        ])
        assert args.input_path == "data.csv"
        assert args.output_file == "output.xlsx"
        assert args.verbose is True
        assert args.config == "config.yaml"

    def test_parse_args_with_short_options(self):
        """Test parsing with short option flags."""
        args = parse_args(["data.csv", "-v", "-o", "result.xlsx"])
        assert args.input_path == "data.csv"
        assert args.verbose is True
        assert args.output_file == "result.xlsx"

    def test_parse_args_with_long_options(self):
        """Test parsing with long option names."""
        args = parse_args([
            "data.csv",
            "--output", "result.xlsx",
            "--verbose"
        ])
        assert args.input_path == "data.csv"
        assert args.output_file == "result.xlsx"
        assert args.verbose is True

    def test_parse_args_empty_input(self):
        """Test parsing with no arguments raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default_level(self, caplog):
        """Test logging setup with default (INFO) level."""
        setup_logging(verbose=False)
        logger = logging.getLogger()

        assert logger.level == logging.INFO

    def test_setup_logging_verbose_mode(self, caplog):
        """Test logging setup with verbose (DEBUG) level."""
        setup_logging(verbose=True)
        logger = logging.getLogger()

        assert logger.level == logging.DEBUG

    def test_setup_logging_format(self):
        """Test that logging format is set correctly."""
        setup_logging(verbose=False)
        logger = logging.getLogger()

        # Check that handlers are configured
        assert len(logger.handlers) > 0

        handler = logger.handlers[0]
        assert handler.formatter is not None


class TestRunApp:
    """Tests for run_app function."""

    @patch('main.logging.getLogger')
    def test_run_app_success(self, mock_logger_class):
        """Test successful app execution."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        args = Mock()
        args.input_path = "data.csv"
        args.output_file = None

        exit_code = run_app(args)

        assert exit_code == 0
        assert mock_logger.info.called

    @patch('main.logging.getLogger')
    def test_run_app_with_output_file(self, mock_logger_class):
        """Test app execution with output file specified."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        args = Mock()
        args.input_path = "data.csv"
        args.output_file = "output.xlsx"

        exit_code = run_app(args)

        assert exit_code == 0

    @patch('main.logging.getLogger')
    def test_run_app_file_not_found(self, mock_logger_class):
        """Test app handles FileNotFoundError correctly."""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        args = Mock()
        args.input_path = "nonexistent.csv"
        args.output_file = None

        # Mock FileNotFoundError in the implementation
        with patch('main.logger', mock_logger):
            # Simulate FileNotFoundError
            mock_logger.info.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                run_app(args)


class TestMain:
    """Tests for main function."""

    @patch('main.parse_args')
    @patch('main.setup_logging')
    @patch('main.run_app')
    @patch('main.sys.exit')
    def test_main_successful_execution(self, mock_exit, mock_run_app,
                                       mock_setup_logging, mock_parse_args):
        """Test main function executes successfully."""
        mock_args = Mock()
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        mock_run_app.return_value = 0

        main()

        mock_parse_args.assert_called_once()
        mock_setup_logging.assert_called_once_with(verbose=False)
        mock_run_app.assert_called_once_with(mock_args)
        mock_exit.assert_called_once_with(0)

    @patch('main.parse_args')
    @patch('main.setup_logging')
    @patch('main.sys.exit')
    def test_main_keyboard_interrupt(self, mock_exit, mock_setup_logging,
                                     mock_parse_args):
        """Test main handles KeyboardInterrupt gracefully."""
        mock_args = Mock()
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        mock_run_app = Mock(side_effect=KeyboardInterrupt())
        with patch('main.run_app', mock_run_app):
            main()

        # Should exit with code 130 for SIGINT
        mock_exit.assert_called_once_with(130)

    @patch('main.parse_args')
    @patch('main.sys.exit')
    def test_main_unhandled_exception(self, mock_exit, mock_parse_args):
        """Test main handles unhandled exceptions."""
        mock_args = Mock()
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args

        mock_setup_logging = Mock(side_effect=Exception("Unexpected error"))
        with patch('main.setup_logging', mock_setup_logging):
            main()

        # Should exit with code 1
        mock_exit.assert_called_once_with(1)
