#!/usr/bin/env python
"""
RAG Pipeline Test Script.

Tests the RAG pipeline and its integration with the evaluation system.
"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_rag.graph.graph import run_rag_pipeline, stream_rag_pipeline
from rag_rag.graph.state import (
    create_initial_state,
    state_to_rag_response_format,
)
from rag_rag.core.config import get_config, RAGConfig


async def test_basic_pipeline():
    """Test basic pipeline execution."""
    print("\n=== Test 1: Basic Pipeline ===")

    result = await run_rag_pipeline(
        query="How to apply for annual leave?",
        enable_thinking=False,
    )

    print(f"Query: {result['query']}")
    print(f"Answer: {result.get('answer', 'N/A')[:100]}")
    print(f"Is Refused: {result.get('is_refused')}")
    print(f"Timing: {result.get('stage_timing', {})}")

    return result


async def test_thinking_mode():
    """Test pipeline with thinking mode enabled."""
    print("\n=== Test 2: Thinking Mode ===")

    result = await run_rag_pipeline(
        query="What are the company policies?",
        enable_thinking=True,
    )

    print(f"Query: {result['query']}")
    print(f"Thinking enabled: True")
    print(f"Answer: {result.get('answer', 'N/A')[:100]}")

    return result


async def test_streaming():
    """Test streaming pipeline execution."""
    print("\n=== Test 3: Streaming ===")

    events = []
    async for event in stream_rag_pipeline(query="Test streaming query"):
        events.append(event)

    print(f"Total events: {len(events)}")
    print(f"Event types: {list(set(type(e).__name__ for e in events))}")

    return events


async def test_eval_integration():
    """Test integration with evaluation system."""
    print("\n=== Test 4: Evaluation System Integration ===")

    # Run pipeline
    result = await run_rag_pipeline(
        query="How do I request PTO?",
        enable_thinking=False,
    )

    # Convert to RAGResponse format
    output = state_to_rag_response_format(result)

    # Import evaluation system's adapter
    try:
        from models.rag_response import RAGResponseAdapter

        rag_response = RAGResponseAdapter.from_langgraph(output, result["query"])

        print(f"RAGResponse created successfully!")
        print(f"  - Query: {rag_response.query}")
        print(f"  - Success: {rag_response.success}")
        print(f"  - Is Refused: {rag_response.is_refused}")
        print(f"  - Answer length: {len(rag_response.final_answer)}")

        if rag_response.query_rewrite:
            print(
                f"  - Query rewritten: {rag_response.query_rewrite.rewritten_query}"
            )

        print("[OK] Evaluation system integration test PASSED")
        return True

    except ImportError as e:
        print(f"[WARN] Could not import evaluation system: {e}")
        return False


async def test_state_creation():
    """Test state creation and validation."""
    print("\n=== Test 5: State Creation ===")

    state = create_initial_state(
        query="Test question",
        conversation_id="test-123",
        agent_id="test-agent",
        enable_thinking=True,
        conversation_history=[
            {"role": "user", "content": "Previous question", "timestamp": "2024-01-01"}
        ],
    )

    print(f"State created with query: {state['query']}")
    print(f"Conversation ID: {state['conversation_id']}")
    print(f"History items: {len(state['conversation_history'])}")
    print(f"Initial errors: {state['errors']}")

    print("[OK] State creation test PASSED")
    return state


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RAG Pipeline Test Suite")
    print("=" * 60)

    results = {}

    try:
        results["basic"] = await test_basic_pipeline()
    except Exception as e:
        print(f"[FAIL] Basic pipeline: {e}")
        results["basic"] = None

    try:
        results["thinking"] = await test_thinking_mode()
    except Exception as e:
        print(f"[FAIL] Thinking mode: {e}")
        results["thinking"] = None

    try:
        results["streaming"] = await test_streaming()
    except Exception as e:
        print(f"[FAIL] Streaming: {e}")
        results["streaming"] = None

    try:
        results["eval"] = await test_eval_integration()
    except Exception as e:
        print(f"[FAIL] Eval integration: {e}")
        results["eval"] = None

    try:
        results["state"] = await test_state_creation()
    except Exception as e:
        print(f"[FAIL] State creation: {e}")
        results["state"] = None

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v is not None and v is not False)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("[OK] All tests PASSED!")
    else:
        print("[WARN] Some tests failed")

    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())