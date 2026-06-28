#!/usr/bin/env python3
"""Demo: System Monitoring + Auto-Remediation."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.pipeline import TDPipeline


def main():
    print("=" * 60)
    print("  TD v2 Demo: System Monitoring")
    print("=" * 60)

    pipeline = TDPipeline(dim=2000)

    print("\n--- Scenario: CPU threshold alert ---")
    decision = pipeline.decide(
        "If CPU exceeds 90 percent for 5 minutes restart the service",
        input_type="natural_language",
        context={
            "constraints": {
                "threshold_exceeded": True,
                "during_maintenance_window": False,
                "service_healthy": False,
                "alert_channel_available": True,
            }
        },
    )
    print(decision.full_trace())

    print("\n--- Scenario: Routine log cleanup ---")
    decision2 = pipeline.decide(
        "Check disk usage daily and delete logs older than 30 days"
    )
    print(decision2.full_trace())

    print("\n--- Scenario: Escalation (unknown task) ---")
    decision3 = pipeline.decide(
        "Design a marketing strategy for a new product launch"
    )
    print(decision3.full_trace())

    print("\n  Demo complete!")


if __name__ == "__main__":
    main()
