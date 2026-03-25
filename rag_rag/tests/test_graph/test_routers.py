"""
Tests for LangGraph routing functions.

Tests the conditional routing logic in the RAG pipeline.
"""

import pytest

from rag_rag.graph.routers import route_after_faq, route_after_refusal


class TestRouteAfterFAQ:
    """Tests for route_after_faq function."""

    def test_route_to_answer_faq_when_matched(self, state_with_faq_matched: dict):
        """When FAQ is matched, should route to answer_faq node."""
        result = route_after_faq(state_with_faq_matched)
        assert result == "answer_faq"

    def test_route_to_query_rewrite_when_not_matched(self, sample_state: dict):
        """When FAQ is not matched, should route to query_rewrite node."""
        sample_state["faq_matched"] = False
        result = route_after_faq(sample_state)
        assert result == "query_rewrite"

    def test_route_to_query_rewrite_when_faq_result_is_none(self, sample_state: dict):
        """When faq_result is None, should route to query_rewrite."""
        sample_state["faq_matched"] = False
        sample_state["faq_result"] = None
        result = route_after_faq(sample_state)
        assert result == "query_rewrite"

    def test_route_to_query_rewrite_default(self, sample_state: dict):
        """When faq_matched is not set (default False), should route to query_rewrite."""
        # Default state has faq_matched=False
        result = route_after_faq(sample_state)
        assert result == "query_rewrite"

    def test_route_to_answer_faq_with_low_confidence_match(self, sample_state: dict):
        """When FAQ is matched (regardless of confidence), should route to answer_faq."""
        sample_state["faq_matched"] = True
        sample_state["faq_result"] = {
            "matched": True,
            "confidence": 0.5,  # Low confidence but still matched
        }
        result = route_after_faq(sample_state)
        assert result == "answer_faq"


class TestRouteAfterRefusal:
    """Tests for route_after_refusal function."""

    def test_route_to_refuse_when_should_refuse(self, state_should_refuse: dict):
        """When should_refuse is True, should route to refuse node."""
        result = route_after_refusal(state_should_refuse)
        assert result == "refuse"

    def test_route_to_generate_when_not_refuse(self, state_with_retrieval_results: dict):
        """When should_refuse is False, should route to generate node."""
        state_with_retrieval_results["should_refuse"] = False
        result = route_after_refusal(state_with_retrieval_results)
        assert result == "generate"

    def test_route_to_generate_default(self, sample_state: dict):
        """When should_refuse is not set (default False), should route to generate."""
        result = route_after_refusal(sample_state)
        assert result == "generate"

    def test_route_to_refuse_with_out_of_domain(self, sample_state: dict):
        """When refusal_type is out_of_domain, should route to refuse."""
        sample_state["should_refuse"] = True
        sample_state["refusal_type"] = "out_of_domain"
        result = route_after_refusal(sample_state)
        assert result == "refuse"

    def test_route_to_refuse_with_low_relevance(self, sample_state: dict):
        """When refusal_type is low_relevance, should route to refuse."""
        sample_state["should_refuse"] = True
        sample_state["refusal_type"] = "low_relevance"
        result = route_after_refusal(sample_state)
        assert result == "refuse"

    def test_route_to_refuse_with_sensitive(self, sample_state: dict):
        """When refusal_type is sensitive, should route to refuse."""
        sample_state["should_refuse"] = True
        sample_state["refusal_type"] = "sensitive"
        result = route_after_refusal(sample_state)
        assert result == "refuse"


class TestRouterEdgeCases:
    """Edge case tests for routing functions."""

    def test_route_after_faq_with_empty_state(self):
        """Test route_after_faq with minimal state."""
        state = {}
        result = route_after_faq(state)
        # Should default to query_rewrite (faq_matched defaults to False)
        assert result == "query_rewrite"

    def test_route_after_refusal_with_empty_state(self):
        """Test route_after_refusal with minimal state."""
        state = {}
        result = route_after_refusal(state)
        # Should default to generate (should_refuse defaults to False)
        assert result == "generate"

    def test_route_after_faq_with_faq_matched_false_but_result_exists(self, sample_state: dict):
        """When faq_matched is False but faq_result exists, should still route to query_rewrite."""
        sample_state["faq_matched"] = False
        sample_state["faq_result"] = {
            "matched": True,
            "answer": "Some answer",
        }
        result = route_after_faq(sample_state)
        assert result == "query_rewrite"