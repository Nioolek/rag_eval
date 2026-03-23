"""RAG Pipeline constants."""

# Stage names
STAGE_INPUT = "input"
STAGE_FAQ_MATCH = "faq_match"
STAGE_QUERY_REWRITE = "query_rewrite"
STAGE_VECTOR_RETRIEVE = "vector_retrieve"
STAGE_FULLTEXT_RETRIEVE = "fulltext_retrieve"
STAGE_GRAPH_RETRIEVE = "graph_retrieve"
STAGE_MERGE = "merge"
STAGE_RERANK = "rerank"
STAGE_BUILD_PROMPT = "build_prompt"
STAGE_REFUSAL_CHECK = "refusal_check"
STAGE_GENERATE = "generate"
STAGE_OUTPUT = "output"
STAGE_ANSWER_FAQ = "answer_faq"
STAGE_REFUSE = "refuse"

# Query rewrite types
REWRITE_TYPE_EXPANSION = "expansion"
REWRITE_TYPE_CLARIFICATION = "clarification"
REWRITE_TYPE_MULTI_TURN = "multi_turn"
REWRITE_TYPE_TRANSLATION = "translation"

# Refusal types
REFUSAL_TYPE_OUT_OF_DOMAIN = "out_of_domain"
REFUSAL_TYPE_SENSITIVE = "sensitive"
REFUSAL_TYPE_LOW_RELEVANCE = "low_relevance"

# Match types
MATCH_TYPE_EXACT = "exact"
MATCH_TYPE_SEMANTIC = "semantic"

# Source types
SOURCE_VECTOR = "vector"
SOURCE_FULLTEXT = "fulltext"
SOURCE_GRAPH = "graph"

# Default values
DEFAULT_TOP_K = 20
DEFAULT_RERANK_TOP_K = 5
DEFAULT_MAX_HISTORY_TURNS = 5
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3

# Thresholds
DEFAULT_FAQ_MATCH_THRESHOLD = 0.85
DEFAULT_OUT_OF_DOMAIN_THRESHOLD = 0.3
DEFAULT_LOW_RELEVANCE_THRESHOLD = 0.2

# Timing keys
TIMING_KEYS = [
    "input_ms",
    "faq_match_ms",
    "query_rewrite_ms",
    "vector_retrieve_ms",
    "fulltext_retrieve_ms",
    "graph_retrieve_ms",
    "merge_ms",
    "rerank_ms",
    "build_prompt_ms",
    "refusal_check_ms",
    "generation_ms",
    "total_ms",
]