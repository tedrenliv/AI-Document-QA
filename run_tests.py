#!/usr/bin/env python3
"""
Test Runner for AI Backend Unit Tests

Provides a convenient CLI for running backend tests with different
configurations (unit-only, integration-only, specific class, etc.).
"""

import sys
import unittest
import argparse
import logging
from io import StringIO


def setup_logging(level=logging.WARNING):
    """Set up logging configuration for tests."""
    logging.basicConfig(
        level=level,
        format='%(levelname)s:%(name)s:%(message)s',
        stream=sys.stdout
    )


def _run_test_suite(test_classes, title, verbosity, quiet):
    """
    Shared helper that loads, runs, and summarises a list of test classes.

    Returns:
        bool: True if all tests passed.
    """
    suite = unittest.TestSuite()
    for cls in test_classes:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    original_stdout = sys.stdout
    if quiet:
        sys.stdout = StringIO()

    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)

    if quiet:
        sys.stdout = original_stdout

    print(f"\n{'='*50}")
    if title:
        print(title)
        print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*50}")

    for label, items in [("FAILURES", result.failures), ("ERRORS", result.errors)]:
        if items:
            print(f"\n{label}:")
            for test, traceback in items:
                print(f"- {test}: {traceback}")

    return result.wasSuccessful()


def _get_unit_test_classes():
    import test_backends
    return [
        test_backends.TestAIBackendInterface,
        test_backends.TestOllamaBackend,
        test_backends.TestGeminiBackend,
        test_backends.TestAIBackendFactory,
        test_backends.TestBackendAvailabilityChecking,
    ]


def _get_integration_test_classes():
    import test_integration
    return [
        test_integration.TestCompleteQAWorkflow,
        test_integration.TestBackendSwitching,
        test_integration.TestErrorScenarios,
        test_integration.TestLoggingConsistency,
        test_integration.TestUIUpdates,
    ]


def run_specific_test_class(test_class_name, verbosity=2):
    """Run tests for a specific test class."""
    import test_backends
    import test_integration

    test_class = getattr(test_backends, test_class_name, None) or getattr(test_integration, test_class_name, None)
    if test_class is None:
        print(f"Error: Test class '{test_class_name}' not found")
        return False

    return _run_test_suite([test_class], None, verbosity, quiet=False)


def main():
    parser = argparse.ArgumentParser(description="Run AI Backend Unit Tests")
    parser.add_argument('--class', '-c', dest='test_class', help='Run tests for a specific class')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Summary only')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--list-classes', action='store_true', help='List available test classes')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    args = parser.parse_args()

    if args.debug:
        setup_logging(logging.DEBUG)
    elif args.verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)

    if args.list_classes:
        print("Available test classes:")
        print("Unit Tests:")
        print("- TestAIBackendInterface")
        print("- TestOllamaBackend")
        print("- TestGeminiBackend")
        print("- TestAIBackendFactory")
        print("- TestBackendAvailabilityChecking")
        print("\nIntegration Tests:")
        print("- TestCompleteQAWorkflow")
        print("- TestBackendSwitching")
        print("- TestErrorScenarios")
        print("- TestLoggingConsistency")
        print("- TestUIUpdates")
        return 0

    verbosity = 0 if args.quiet else (2 if args.verbose else 1)

    try:
        if args.test_class:
            success = run_specific_test_class(args.test_class, verbosity)
        elif args.unit_only:
            success = _run_test_suite(_get_unit_test_classes(), "UNIT TESTS SUMMARY", verbosity, args.quiet)
        elif args.integration_only:
            success = _run_test_suite(_get_integration_test_classes(), "INTEGRATION TESTS SUMMARY", verbosity, args.quiet)
        else:
            classes = _get_unit_test_classes() + _get_integration_test_classes()
            success = _run_test_suite(classes, None, verbosity, args.quiet)
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
