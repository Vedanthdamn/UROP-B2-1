"""
Test suite for federated training experiments script.

This test suite validates the run_federated_experiments.py script:
1. Script execution with minimal rounds
2. Output file generation (JSON and Markdown)
3. FedAvg and FedProx strategies
4. Differential privacy integration

Usage:
    python test_federated_experiments.py
"""

import sys
import os
import json
import subprocess


def test_script_execution_fedavg():
    """Test script execution with FedAvg strategy."""
    print("Test 1: Script Execution with FedAvg")
    
    # Run script with minimal rounds for testing
    result = subprocess.run(
        [
            sys.executable,
            "run_federated_experiments.py",
            "--num-rounds", "2",
            "--num-clients", "5",
            "--strategy", "fedavg"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert result.returncode == 0, f"Script failed with exit code {result.returncode}"
    print("  ✓ Script executed successfully with FedAvg")


def test_output_files_exist():
    """Test that output files are generated."""
    print("Test 2: Output Files Generation")
    
    # Check if training_history.json exists
    assert os.path.exists("logs/training_history.json"), "training_history.json not found"
    print("  ✓ training_history.json exists")
    
    # Check if training_summary.md exists
    assert os.path.exists("logs/training_summary.md"), "training_summary.md not found"
    print("  ✓ training_summary.md exists")


def test_json_structure():
    """Test the structure of training_history.json."""
    print("Test 3: JSON Structure Validation")
    
    with open("logs/training_history.json", "r") as f:
        data = json.load(f)
    
    # Check top-level keys
    assert "experiment_config" in data, "experiment_config not found"
    assert "timestamp" in data, "timestamp not found"
    assert "rounds" in data, "rounds not found"
    print("  ✓ JSON has required top-level keys")
    
    # Check experiment_config
    config = data["experiment_config"]
    assert "num_clients" in config, "num_clients not in config"
    assert "num_rounds" in config, "num_rounds not in config"
    assert "strategy" in config, "strategy not in config"
    assert "model" in config, "model not in config"
    print("  ✓ experiment_config has required fields")
    
    # Check rounds data
    assert len(data["rounds"]) > 0, "No rounds data found"
    round_data = data["rounds"][0]
    assert "round" in round_data, "round not in round_data"
    assert "global_loss" in round_data, "global_loss not in round_data"
    assert "global_accuracy" in round_data, "global_accuracy not in round_data"
    assert "participating_clients" in round_data, "participating_clients not in round_data"
    print("  ✓ Round data has required fields")


def test_markdown_structure():
    """Test the structure of training_summary.md."""
    print("Test 4: Markdown Structure Validation")
    
    with open("logs/training_summary.md", "r") as f:
        content = f.read()
    
    # Check for required sections
    assert "# Federated Training Experiment Summary" in content, "Title not found"
    assert "## Experiment Configuration" in content, "Configuration section not found"
    assert "## Per-Round Training Metrics" in content, "Metrics section not found"
    assert "## Training Summary" in content, "Training summary not found"
    assert "## Privacy Guarantees" in content, "Privacy guarantees not found"
    assert "## Client Participation Summary" in content, "Participation summary not found"
    print("  ✓ Markdown has required sections")


def test_script_execution_fedprox():
    """Test script execution with FedProx strategy."""
    print("Test 5: Script Execution with FedProx")
    
    # Run script with FedProx strategy
    result = subprocess.run(
        [
            sys.executable,
            "run_federated_experiments.py",
            "--num-rounds", "2",
            "--num-clients", "5",
            "--strategy", "fedprox",
            "--proximal-mu", "0.1"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert result.returncode == 0, f"Script failed with exit code {result.returncode}"
    print("  ✓ Script executed successfully with FedProx")
    
    # Verify strategy in output
    with open("logs/training_history.json", "r") as f:
        data = json.load(f)
    
    assert data["experiment_config"]["strategy"] == "fedprox", "Strategy mismatch"
    assert data["experiment_config"]["proximal_mu"] == 0.1, "Proximal mu mismatch"
    print("  ✓ FedProx configuration verified")


def test_script_execution_with_dp():
    """Test script execution with differential privacy."""
    print("Test 6: Script Execution with Differential Privacy")
    
    # Run script with DP enabled
    result = subprocess.run(
        [
            sys.executable,
            "run_federated_experiments.py",
            "--num-rounds", "2",
            "--num-clients", "5",
            "--strategy", "fedavg",
            "--use-dp",
            "--dp-epsilon", "1.0",
            "--dp-delta", "1e-5"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    assert result.returncode == 0, f"Script failed with exit code {result.returncode}"
    print("  ✓ Script executed successfully with DP")
    
    # Verify DP in output
    with open("logs/training_history.json", "r") as f:
        data = json.load(f)
    
    assert data["experiment_config"]["use_dp"] == True, "DP not enabled"
    assert data["experiment_config"]["dp_epsilon"] == 1.0, "DP epsilon mismatch"
    print("  ✓ Differential privacy configuration verified")


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("FEDERATED TRAINING EXPERIMENTS TEST SUITE")
    print("=" * 70 + "\n")
    
    tests = [
        test_script_execution_fedavg,
        test_output_files_exist,
        test_json_structure,
        test_markdown_structure,
        test_script_execution_fedprox,
        test_script_execution_with_dp,
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")
    
    if failed > 0:
        print("Some tests failed!")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_tests()
