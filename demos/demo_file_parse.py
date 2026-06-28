#!/usr/bin/env python3
"""Demo: File Processing Pipeline."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.pipeline import TDPipeline


def main():
    print("=" * 60)
    print("  TD v2 Demo: File Processing")
    print("=" * 60)

    pipeline = TDPipeline(dim=2000)

    print("\n--- Scenario: CSV Parse + Validate ---")
    decision = pipeline.decide(
        "Parse the CSV file and validate columns match the expected schema"
    )
    print(decision.full_trace())

    print("\n--- Scenario: JSON to CSV Transform ---")
    decision2 = pipeline.decide(
        "Convert JSON to CSV with specific column mapping"
    )
    print(decision2.full_trace())

    print("\n  Demo complete!")


if __name__ == "__main__":
    main()
