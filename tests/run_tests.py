#!/usr/bin/env python
import os
import sys
import unittest
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_tests(test_modules=None, generate_report=False):
    """Run the test suite
    
    Args:
        test_modules (list): Specific test modules to run (default: all)
        generate_report (bool): Whether to generate an HTML report
    """
    # Discover and load all tests
    loader = unittest.TestLoader()
    
    if test_modules:
        # Load specific test modules
        suite = unittest.TestSuite()
        for module_name in test_modules:
            try:
                tests = loader.loadTestsFromName(f"tests.{module_name}")
                suite.addTests(tests)
            except ImportError:
                print(f"Error: Test module 'tests.{module_name}' not found.")
                sys.exit(1)
    else:
        # Load all tests from the tests directory
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(start_dir)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate HTML report if requested
    if generate_report:
        try:
            from unittest import TextTestRunner
            from unittest.runner import TextTestResult
            import HtmlTestRunner
            
            # Create reports directory
            reports_dir = project_root / "test_reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Generate timestamp for report name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"finance_rag_test_report_{timestamp}.html"
            
            # Run tests with HTML test runner
            html_runner = HtmlTestRunner.HTMLTestRunner(
                output=str(reports_dir),
                report_name=f"finance_rag_test_report_{timestamp}",
                combine_reports=True,
                template_args={
                    "title": "Finance RAG Tests",
                    "heading": "Finance RAG Test Results"
                }
            )
            html_runner.run(suite)
            
            print(f"\nHTML test report generated at: {report_file}")
            
        except ImportError:
            print("\nHTML report generation requires HtmlTestRunner.")
            print("Install with: pip install html-testRunner")
    
    # Return success status
    return result.wasSuccessful()

def list_test_modules():
    """List all available test modules"""
    start_dir = os.path.dirname(os.path.abspath(__file__))
    modules = []
    
    for file in os.listdir(start_dir):
        if file.startswith("test_") and file.endswith(".py"):
            module_name = file[:-3]  # Remove .py extension
            modules.append(module_name)
    
    return modules

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Finance RAG tests")
    parser.add_argument(
        "-m", "--modules",
        nargs="+",
        help="Specific test modules to run (without the 'tests.' prefix)"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all available test modules"
    )
    parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="Generate HTML test report"
    )
    
    args = parser.parse_args()
    
    if args.list:
        modules = list_test_modules()
        print("Available test modules:")
        for module in sorted(modules):
            print(f"  - {module}")
        sys.exit(0)
    
    # Run the tests
    success = run_tests(args.modules, args.report)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 