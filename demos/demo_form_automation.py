#!/usr/bin/env python3
"""Demo: Web Form Automation.

Shows TD v2 processing a web form automation task:
NL input → perceive → route → (optionally) validate → decide.

Run: python demos/demo_form_automation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.pipeline import TDPipeline


SAMPLE_HTML = """
<html>
<body>
  <form id="contact-form" method="post">
    <input type="text" name="name" placeholder="Your name" required>
    <input type="email" name="email" placeholder="Your email" required>
    <textarea name="message" placeholder="Message" required></textarea>
    <button type="submit">Send</button>
  </form>
</body>
</html>
"""


def main():
    print("=" * 60)
    print("  TD v2 Demo: Web Form Automation")
    print("=" * 60)

    # Initialize pipeline
    print("\nInitializing TD v2 pipeline...")
    pipeline = TDPipeline(dim=2000)
    print("  ✓ HDC vocabulary loaded")
    print("  ✓ CA Reservoir ready")
    print("  ✓ MHN memory initialized")
    print("  ✓ Hierarchical Router ready")
    print("  ✓ Z3 Bridge ready")

    # Show a decision without any prior memory
    print("\n" + "-" * 60)
    print("SCENARIO 1: First encounter (no memory)")
    print("-" * 60)
    decision = pipeline.decide(
        "Fill out the contact form with name=Alice email=alice@test.com",
        input_type="natural_language",
        dom_html=SAMPLE_HTML,
    )
    print(decision.full_trace())

    # Learn from a successful execution
    print("\n" + "-" * 60)
    print("SCENARIO 2: Learning from execution")
    print("-" * 60)
    pipeline.learn(
        "Fill out the contact form with name and email",
        [
            {"action": "click", "target": "name_field"},
            {"action": "type", "target": "name_field", "value": "Alice"},
            {"action": "click", "target": "email_field"},
            {"action": "type", "target": "email_field", "value": "alice@test.com"},
            {"action": "click", "target": "submit_button"},
        ],
        "success",
        {"domain": "Web", "task_type": "Form"},
    )
    print(f"  ✓ Learned pattern. MHN now has {len(pipeline.mhn)} patterns.")

    # Now encounter a similar task — should retrieve the learned pattern
    print("\n" + "-" * 60)
    print("SCENARIO 3: Similar task (with memory)")
    print("-" * 60)
    decision2 = pipeline.decide(
        "Fill out the contact form with name=Bob email=bob@test.com",
        input_type="natural_language",
        dom_html=SAMPLE_HTML,
    )
    print(decision2.full_trace())

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
