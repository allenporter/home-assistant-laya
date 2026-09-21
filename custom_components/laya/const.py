"""Constants for laya."""

DOMAIN = "laya"

# Configuration keys
CONF_DEVICE = "device"
CONF_STRATEGY = "strategy"
CONF_FALLBACK_AGENT = "fallback_agent"
CONF_COMPOUND_THRESHOLD = "compound_threshold"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"
CONF_IDLE_TIMEOUT = "idle_timeout"

CONF_DOMAIN_FILTER_MODE = "domain_filter_mode"
CONF_RETRIEVER_TYPE = "retriever_type"

DEFAULT_DEVICE = "auto"
DEFAULT_COMPOUND_THRESHOLD = 0.50
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_IDLE_TIMEOUT = 0.0
DEFAULT_DOMAIN_FILTER_MODE = "none"
DEFAULT_RETRIEVER_TYPE = "lexical"

# Performance & cardinality limits
MAX_OPTIONS_PER_QUESTION = 20
