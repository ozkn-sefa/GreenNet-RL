import pytest
import sys
from mininet_scripts import test_mininet_experiments


def test_mininet_script_handles_missing_mininet_gracefully(monkeypatch):
    monkeypatch.setattr(test_mininet_experiments, "MININET_AVAILABLE", False)
    res = test_mininet_experiments.run_mininet_suite()
    assert res["status"] == "skipped"
    assert "not available" in res["reason"]
