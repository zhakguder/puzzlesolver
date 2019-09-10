#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `puzzlesolver` package."""


import unittest
from click.testing import CliRunner

from puzzlesolver import puzzlesolver
from puzzlesolver import cli

class TestPuzzlesolver(unittest.TestCase):
    """Tests for `puzzlesolver` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_puzzlesolver_solve_returns_something(self):
        """Test that puzzlesolver returns something"""
        self.assertIsNotNone (puzzlesolver.solve())

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'puzzlesolver.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
