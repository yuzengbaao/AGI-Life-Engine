#!/usr/bin/env python3
"""
Test runner script for Data Processing Tool.

This script provides convenient commands to run tests with various options.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a command and display its output.

    Args:
        cmd: Command to run as a list of strings
        description: Description of what the command does

    Returns:
        Exit code from the command
    """
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode


def main():
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test runner for Data Processing Tool"
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=["all", "unit", "coverage", "fast", "install"],
        default="all",
        help="Test target to run"
    )

    args = parser.parse_args()

    # Install development dependencies
    if args.target == "install":
        print("\n📦 Installing development dependencies...")
        return run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"],
            "Installing Development Dependencies"
        )

    # Run all tests
    if args.target == "all":
        print("\n🧪 Running all tests with coverage...")
        return run_command(
            [sys.executable, "-m", "pytest", "-v", "--cov=.", "--cov-report=html"],
            "Running All Tests with Coverage"
        )

    # Run unit tests only
    if args.target == "unit":
        print("\n🔬 Running unit tests...")
        return run_command(
            [sys.executable, "-m", "pytest", "-v", "-m", "unit"],
            "Running Unit Tests"
        )

    # Run coverage report
    if args.target == "coverage":
        print("\n📊 Generating coverage report...")
        return run_command(
            [
                sys.executable, "-m", "pytest",
                "--cov=.",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--html=htmlcov/index.html"
            ],
            "Generating Coverage Report"
        )

    # Run fast tests only
    if args.target == "fast":
        print("\n⚡ Running fast tests...")
        return run_command(
            [sys.executable, "-m", "pytest", "-v", "-m", "not slow"],
            "Running Fast Tests"
        )


if __name__ == "__main__":
    sys.exit(main())
