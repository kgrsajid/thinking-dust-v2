#!/usr/bin/env python3
"""Demo: API Workflow Execution.

Shows TD v2 processing an API orchestration task.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.pipeline import TDPipeline


def main():
    print("=" * 60)
    print("  TD v2 Demo: API Workflow Execution")
    print("=" * 60)

    pipeline = TDPipeline(dim=2000)

    print("\n--- Scenario: Sequential API calls ---")
    decision = pipeline.decide(
        "Fetch user profile from API then fetch their orders"
    )
    print(decision.full_trace())

    print("\n--- Scenario: Error handling ---")
    decision2 = pipeline.decide(
        "Handle 404 errors and log the missing endpoints"
    )
    print(decision2.full_trace())

    print("\n--- Scenario: Retry logic ---")
    decision3 = pipeline.decide(
        "Retry failed API calls up to 3 times with exponential backoff"
    )
    print(decision3.full_trace())

    print("\n  Demo complete!")


if __name__ == "__main__":
    main()
