"""Constants for laya."""

DOMAIN = "laya"

# Configuration keys
CONF_DEVICE = "device"
CONF_STRATEGY = "strategy"
CONF_FALLBACK_AGENT = "fallback_agent"
CONF_COMPOUND_THRESHOLD = "compound_threshold"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"
CONF_IDLE_TIMEOUT = "idle_timeout"

DEFAULT_DEVICE = "auto"
DEFAULT_COMPOUND_THRESHOLD = 0.65
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_IDLE_TIMEOUT = 0.0

# Strategies
STRATEGY_SPECULATIVE_FAN_OUT = "speculative_fan_out"
DEFAULT_STRATEGY = STRATEGY_SPECULATIVE_FAN_OUT

# Performance & cardinality limits
MAX_OPTIONS_PER_QUESTION = 20
