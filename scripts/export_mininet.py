"""
CLI Script to export Mininet & Ryu controller files.
Usage:
    python scripts/export_mininet.py --out mininet_scripts
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sdn.mininet_exporter import MininetExporter


def main():
    print("=" * 60)
    print(" [*] MININET & RYU SDN SCRIPTS EXPORTER")
    print("=" * 60)
    MininetExporter.export_all("mininet_scripts")


if __name__ == "__main__":
    main()
