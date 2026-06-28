"""Predefined constraint schemas for common agent domains.

These schemas define the hard constraints that Z3 validates against.
Each schema is a dict of constraint key → expected value.
"""

# Web Form Validation
WEB_FORM_CONSTRAINTS = {
    "submit_visible": True,
    "required_fields_filled": True,
    "captcha_present": False,
    "form_visible": True,
    "submit_enabled": True,
}

# API Sequential Call Constraints
API_SEQUENTIAL_CONSTRAINTS = {
    "auth_token_present": True,
    "rate_limit_exceeded": False,
    "endpoint_available": True,
    "timeout_exceeded": False,
}

# File Parse Constraints
FILE_PARSE_CONSTRAINTS = {
    "file_exists": True,
    "file_readable": True,
    "schema_valid": True,
    "file_not_empty": True,
}

# Monitor Threshold Constraints
MONITOR_THRESHOLD_CONSTRAINTS = {
    "threshold_exceeded": True,
    "during_maintenance_window": False,
    "service_healthy": False,  # If threshold exceeded, service is unhealthy
    "alert_channel_available": True,
}
