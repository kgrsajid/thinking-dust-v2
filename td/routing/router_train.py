"""Training utilities for the hierarchical ternary router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..perception.hdc import ConceptVocabulary, build_default_vocabulary
from ..perception.nl_parser import NLParser
from .router_a import RouterA, DOMAINS
from .router_b import RouterB, TASK_TYPES
from .router_c import RouterC, STRATEGIES
from .hierarchical_router import HierarchicalRouter


# ---------------------------------------------------------------------------
# Synthetic Training Data
# ---------------------------------------------------------------------------

TRAINING_EXAMPLES = [
    # Web / Form (more examples for better coverage)
    ("Click the submit button on the login form", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Fill out the contact form with name and email", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Enter username and password then click login", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Type the search query and hit enter", "Web", "Form", "MEMORY_ONLY"),
    ("Select option from dropdown and submit", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Check the checkbox to agree to terms and conditions", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Fill in the registration form with user details", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Enter credit card information in the checkout form", "Web", "Form", "MEMORY_THEN_VALIDATE"),
    ("Type email address and click subscribe button", "Web", "Form", "MEMORY_ONLY"),
    ("Input password confirmation and submit", "Web", "Form", "MEMORY_THEN_VALIDATE"),

    # Web / Navigation (more examples)
    ("Click the next page button", "Web", "Navigation", "MEMORY_ONLY"),
    ("Scroll down to see more content", "Web", "Navigation", "MEMORY_ONLY"),
    ("Navigate back to the home page", "Web", "Navigation", "MEMORY_ONLY"),
    ("Click on the about us link", "Web", "Navigation", "MEMORY_ONLY"),
    ("Go to the dashboard tab", "Web", "Navigation", "MEMORY_ONLY"),
    ("Scroll up to the top of the page", "Web", "Navigation", "MEMORY_ONLY"),
    ("Click the breadcrumb link to go back", "Web", "Navigation", "MEMORY_ONLY"),
    ("Navigate to the settings page", "Web", "Navigation", "MEMORY_ONLY"),
    ("Open the user profile page", "Web", "Navigation", "MEMORY_ONLY"),
    ("Click the logo to return home", "Web", "Navigation", "MEMORY_ONLY"),

    # Web / Extraction
    ("Extract all product prices from the page", "Web", "Extraction", "MEMORY_ONLY"),
    ("Get the text content of the main heading", "Web", "Extraction", "MEMORY_ONLY"),
    ("Scrape table data from the results page", "Web", "Extraction", "MEMORY_ONLY"),
    ("Extract user reviews from the listing", "Web", "Extraction", "MEMORY_ONLY"),
    ("Get all image URLs from the gallery", "Web", "Extraction", "MEMORY_ONLY"),
    ("Copy the phone number from the contact page", "Web", "Extraction", "MEMORY_ONLY"),
    ("Read the article title and publication date", "Web", "Extraction", "MEMORY_ONLY"),

    # Web / Interaction
    ("Hover over the menu to reveal submenu", "Web", "Interaction", "MEMORY_ONLY"),
    ("Drag and drop the file into the upload zone", "Web", "Interaction", "MEMORY_THEN_VALIDATE"),
    ("Click the checkbox to enable dark mode", "Web", "Interaction", "MEMORY_ONLY"),
    ("Toggle the switch to turn on notifications", "Web", "Interaction", "MEMORY_ONLY"),
    ("Click the play button on the video player", "Web", "Interaction", "MEMORY_ONLY"),
    ("Drag the slider to adjust volume", "Web", "Interaction", "MEMORY_ONLY"),

    # API / Sequential (more examples)
    ("Fetch user profile from API then fetch their orders", "API", "Sequential", "MEMORY_THEN_VALIDATE"),
    ("Call the auth API to get a token then use it for the next request", "API", "Sequential", "MEMORY_THEN_VALIDATE"),
    ("Get data from endpoint A, transform it, send to endpoint B", "API", "Sequential", "MEMORY_THEN_VALIDATE"),
    ("Fetch the list, then fetch details for each item", "API", "Sequential", "MEMORY_THEN_VALIDATE"),
    ("Query the database and return results as JSON", "API", "Sequential", "MEMORY_ONLY"),
    ("Login via API then fetch the dashboard data", "API", "Sequential", "MEMORY_THEN_VALIDATE"),
    ("Get authentication token then call protected endpoint", "API", "Sequential", "MEMORY_THEN_VALIDATE"),
    ("Fetch paginated results until all data is retrieved", "API", "Sequential", "MEMORY_THEN_VALIDATE"),

    # API / Parallel
    ("Fetch data from three APIs simultaneously", "API", "Parallel", "MEMORY_THEN_VALIDATE"),
    ("Make concurrent requests to all endpoints", "API", "Parallel", "MEMORY_THEN_VALIDATE"),
    ("Batch fetch user details for multiple IDs", "API", "Parallel", "MEMORY_ONLY"),
    ("Call multiple microservices in parallel and merge results", "API", "Parallel", "MEMORY_THEN_VALIDATE"),
    ("Send async requests to all webhook URLs", "API", "Parallel", "MEMORY_ONLY"),
    ("Query multiple data sources concurrently", "API", "Parallel", "MEMORY_THEN_VALIDATE"),

    # API / ErrorHandling
    ("Handle 404 errors and log the missing endpoints", "API", "ErrorHandling", "MEMORY_THEN_VALIDATE"),
    ("Catch network timeouts and return cached data", "API", "ErrorHandling", "MEMORY_THEN_VALIDATE"),
    ("If API returns error, fall back to default values", "API", "ErrorHandling", "MEMORY_THEN_VALIDATE"),
    ("Detect rate limiting and reduce request frequency", "API", "ErrorHandling", "MEMORY_ONLY"),
    ("Parse error response and extract error codes", "API", "ErrorHandling", "MEMORY_ONLY"),

    # API / Retry
    ("Retry failed API calls up to 3 times", "API", "Retry", "MEMORY_THEN_VALIDATE"),
    ("Implement exponential backoff for 5xx errors", "API", "Retry", "MEMORY_THEN_VALIDATE"),
    ("Retry with different parameters on timeout", "API", "Retry", "MEMORY_THEN_VALIDATE"),
    ("Reconnect to the websocket on disconnect", "API", "Retry", "MEMORY_ONLY"),

    # File / Parse (more examples)
    ("Parse the CSV file and extract all rows", "File", "Parse", "MEMORY_THEN_VALIDATE"),
    ("Read the JSON configuration file", "File", "Parse", "MEMORY_ONLY"),
    ("Load the YAML file and validate its structure", "File", "Parse", "MEMORY_THEN_VALIDATE"),
    ("Parse XML data and extract attribute values", "File", "Parse", "MEMORY_ONLY"),
    ("Read the TSV file and convert to list of dicts", "File", "Parse", "MEMORY_ONLY"),
    ("Import data from Excel spreadsheet", "File", "Parse", "MEMORY_THEN_VALIDATE"),
    ("Parse the log file for error messages", "File", "Parse", "MEMORY_ONLY"),
    ("Read the properties file and extract key-value pairs", "File", "Parse", "MEMORY_ONLY"),

    # File / Transform
    ("Convert JSON to CSV with specific column mapping", "File", "Transform", "MEMORY_ONLY"),
    ("Transform the data to match the new schema", "File", "Transform", "MEMORY_THEN_VALIDATE"),
    ("Merge multiple CSV files into one", "File", "Transform", "MEMORY_THEN_VALIDATE"),
    ("Filter rows based on column values", "File", "Transform", "MEMORY_ONLY"),
    ("Normalize the date formats across all records", "File", "Transform", "MEMORY_ONLY"),

    # File / Validate
    ("Validate the CSV columns match the expected schema", "File", "Validate", "MEMORY_THEN_VALIDATE"),
    ("Check for null values in required fields", "File", "Validate", "MEMORY_THEN_VALIDATE"),
    ("Verify all email addresses are valid format", "File", "Validate", "MEMORY_THEN_VALIDATE"),
    ("Validate that IDs are unique in the dataset", "File", "Validate", "MEMORY_THEN_VALIDATE"),

    # File / Generate
    ("Generate a CSV report from the data", "File", "Generate", "MEMORY_ONLY"),
    ("Create a summary JSON file with statistics", "File", "Generate", "MEMORY_ONLY"),
    ("Export the filtered results to a new file", "File", "Generate", "MEMORY_ONLY"),
    ("Write the configuration to a YAML file", "File", "Generate", "MEMORY_ONLY"),

    # Monitor / Threshold (more examples)
    ("If CPU exceeds 90 percent for 5 minutes restart the service", "Monitor", "Threshold", "MEMORY_THEN_VALIDATE"),
    ("Alert when memory usage goes above 85 percent", "Monitor", "Threshold", "MEMORY_ONLY"),
    ("Trigger notification when disk is 95 percent full", "Monitor", "Threshold", "MEMORY_ONLY"),
    ("Restart nginx if response time exceeds 2 seconds", "Monitor", "Threshold", "MEMORY_THEN_VALIDATE"),
    ("Scale up if queue length exceeds 1000 messages", "Monitor", "Threshold", "MEMORY_THEN_VALIDATE"),
    ("Send alert when database connection pool is exhausted", "Monitor", "Threshold", "MEMORY_ONLY"),
    ("Restart service when health check fails 3 times", "Monitor", "Threshold", "MEMORY_THEN_VALIDATE"),

    # Monitor / LogAnalysis
    ("Analyze error logs from the last hour", "Monitor", "LogAnalysis", "MEMORY_ONLY"),
    ("Find recurring exceptions in the application logs", "Monitor", "LogAnalysis", "MEMORY_ONLY"),
    ("Parse nginx access logs for 500 errors", "Monitor", "LogAnalysis", "MEMORY_ONLY"),
    ("Detect anomalies in system event logs", "Monitor", "LogAnalysis", "MEMORY_THEN_VALIDATE"),
    ("Summarize warning messages from the log file", "Monitor", "LogAnalysis", "MEMORY_ONLY"),

    # Monitor / AlertRouting
    ("Send critical alerts to the on-call engineer", "Monitor", "AlertRouting", "MEMORY_ONLY"),
    ("Route database warnings to the DBA team", "Monitor", "AlertRouting", "MEMORY_ONLY"),
    ("Notify Slack channel for deployment failures", "Monitor", "AlertRouting", "MEMORY_ONLY"),
    ("Email the admin if SSL certificate expires soon", "Monitor", "AlertRouting", "MEMORY_ONLY"),

    # Monitor / Routine
    ("Check disk usage daily and delete logs older than 30 days", "Monitor", "Routine", "MEMORY_ONLY"),
    ("Run health check every 5 minutes", "Monitor", "Routine", "MEMORY_ONLY"),
    ("Backup the database every night at 2am", "Monitor", "Routine", "MEMORY_ONLY"),
    ("Clean up temporary files every hour", "Monitor", "Routine", "MEMORY_ONLY"),

    # Unknown / Escalate (more examples)
    ("Book a flight to Istanbul with 500 dollar budget morning departure", "Unknown", "Complex", "ESCALATE"),
    ("Plan a 3-day trip itinerary for Paris", "Unknown", "Complex", "ESCALATE"),
    ("Write a poem about artificial intelligence", "Unknown", "Novel", "ESCALATE"),
    ("Prove this sorting algorithm is correct", "Unknown", "Proof", "ESCALATE"),
    ("Design a marketing strategy for a new product", "Unknown", "Complex", "ESCALATE"),
    ("The input is ambiguous and unclear", "Unknown", "Ambiguous", "ESCALATE"),
    ("Translate this document to French", "Unknown", "Novel", "ESCALATE"),
    ("Generate creative names for a coffee shop", "Unknown", "Novel", "ESCALATE"),
    ("Negotiate the best deal with the vendor", "Unknown", "Complex", "ESCALATE"),
    ("Analyze the sentiment of customer feedback", "Unknown", "Complex", "ESCALATE"),
    ("Write a song about the ocean", "Unknown", "Novel", "ESCALATE"),
    ("Compose a resume for a software engineer", "Unknown", "Novel", "ESCALATE"),
    ("Decide what to cook for dinner tonight", "Unknown", "Ambiguous", "ESCALATE"),
    ("Recommend a movie based on my preferences", "Unknown", "Complex", "ESCALATE"),
    ("Explain quantum computing to a five year old", "Unknown", "Novel", "ESCALATE"),
    ("Create a lesson plan for teaching fractions", "Unknown", "Complex", "ESCALATE"),
    ("Debate whether AI will replace human jobs", "Unknown", "Complex", "ESCALATE"),
    ("Summarize the plot of Romeo and Juliet", "Unknown", "Novel", "ESCALATE"),
    ("Brainstorm ideas for a mobile app", "Unknown", "Complex", "ESCALATE"),
    ("Review this legal contract for issues", "Unknown", "Proof", "ESCALATE"),
]


def generate_training_data(vocab: ConceptVocabulary) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate encoded training data from synthetic examples.

    Returns:
        Tuple of (hdc_vectors, domain_labels, tasktype_labels, strategy_labels)
        - hdc_vectors: np.ndarray (N, 10_000)
        - domain_labels: np.ndarray (N,) int indices
        - tasktype_labels: np.ndarray (N,) int indices
        - strategy_labels: np.ndarray (N,) int indices
    """
    parser = NLParser(vocab)
    vectors = []
    domains = []
    task_types_list = []
    strategies_list = []

    for text, domain, task_type, strategy in TRAINING_EXAMPLES:
        vec = parser.parse(text)
        vectors.append(vec)
        domains.append(DOMAINS.index(domain))

        domain_tasks = TASK_TYPES[domain]
        task_idx = domain_tasks.index(task_type) if task_type in domain_tasks else 0
        task_types_list.append(task_idx)

        strategies_list.append(STRATEGIES.index(strategy))

    return (
        np.stack(vectors),
        np.array(domains, dtype=np.int64),
        np.array(task_types_list, dtype=np.int64),
        np.array(strategies_list, dtype=np.int64),
    )


def train_router(
    vocab: ConceptVocabulary | None = None,
    epochs: int = 100,
    lr: float = 5e-3,
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """Train all three routers on synthetic data.

    Args:
        vocab: Concept vocabulary. Builds default if None.
        epochs: Training epochs per router.
        lr: Learning rate.
        batch_size: Mini-batch size.
        verbose: Print progress.

    Returns:
        Dict with training metrics for each router.
    """
    if vocab is None:
        vocab = build_default_vocabulary()

    hdc_vecs, domain_labels, tasktype_labels, strategy_labels = (
        generate_training_data(vocab)
    )

    X = torch.from_numpy(hdc_vecs.astype(np.float32))

    metrics = {}

    # --- Train Router A (Domain) ---
    if verbose:
        print("Training Router A (Domain Detector)...")
    router_a = RouterA(input_dim=vocab.dim)
    optimizer_a = optim.Adam(router_a.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset_a = TensorDataset(X, torch.from_numpy(domain_labels))
    loader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        router_a.train()
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in loader_a:
            optimizer_a.zero_grad()
            out = router_a(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer_a.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(xb)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/total:.4f} acc={correct/total:.3f}")

    metrics["router_a"] = {"final_loss": total_loss / total, "final_acc": correct / total}

    # --- Train Router B (per domain) ---
    metrics["router_b"] = {}
    routers_b: dict[str, RouterB] = {}
    for domain in DOMAINS:
        mask = domain_labels == DOMAINS.index(domain)
        if mask.sum() == 0:
            continue
        X_d = X[mask]
        y_d = torch.from_numpy(tasktype_labels[mask])

        if verbose:
            print(f"Training Router B ({domain})... ({len(X_d)} examples)")

        router_b = RouterB(domain, input_dim=vocab.dim)
        routers_b[domain] = router_b
        optimizer_b = optim.Adam(router_b.parameters(), lr=lr)

        if len(X_d) <= batch_size:
            # Single batch
            for epoch in range(epochs):
                optimizer_b.zero_grad()
                out = router_b(X_d)
                loss = criterion(out, y_d)
                loss.backward()
                optimizer_b.step()
        else:
            dataset_b = TensorDataset(X_d, y_d)
            loader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                router_b.train()
                for xb, yb in loader_b:
                    optimizer_b.zero_grad()
                    out = router_b(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer_b.step()

        with torch.no_grad():
            router_b.eval()
            preds = router_b(X_d).argmax(dim=1)
            acc = (preds == y_d).float().mean().item()
        metrics["router_b"][domain] = {"final_acc": acc}

    # --- Train Router C (Strategy) ---
    if verbose:
        print("Training Router C (Strategy Selector)...")
    router_c = RouterC(input_dim=vocab.dim)
    optimizer_c = optim.Adam(router_c.parameters(), lr=lr)

    dataset_c = TensorDataset(X, torch.from_numpy(strategy_labels))
    loader_c = DataLoader(dataset_c, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        router_c.train()
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in loader_c:
            optimizer_c.zero_grad()
            out = router_c(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer_c.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(xb)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/total:.4f} acc={correct/total:.3f}")

    metrics["router_c"] = {"final_loss": total_loss / total, "final_acc": correct / total}

    if verbose:
        print("\nTraining complete!")
        print(f"  Router A accuracy: {metrics['router_a']['final_acc']:.3f}")
        for d, m in metrics["router_b"].items():
            print(f"  Router B ({d}) accuracy: {m['final_acc']:.3f}")
        print(f"  Router C accuracy: {metrics['router_c']['final_acc']:.3f}")

    return {
        "metrics": metrics,
        "router_a": router_a,
        "routers_b": routers_b,
        "router_c": router_c,
    }
