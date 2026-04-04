#!/usr/bin/env python3
"""
Model Evaluation: Qwen3-Coder-Next-AWQ-4bit

Solo evaluation of a model served via vLLM OpenAI-compatible endpoint.

Tests: code generation, reasoning, scaffolding/tool-use, general chat.
Metrics: TTFT, tokens/sec, total latency, pass/fail.

Usage: python3 ~/model_ab_eval.py [--categories code,reasoning,scaffold,chat]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

ENDPOINTS = {
    "model": {
        "base_url": "http://localhost:8000/v1",
        "label": "Qwen3-Coder-Next-AWQ-4bit",
    },
}

DEFAULT_TIMEOUT = 120
LONG_CTX_TIMEOUT = 600

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_model(base_url: str) -> str:
    """Auto-detect the model name from the /v1/models endpoint."""
    resp = requests.get(f"{base_url}/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]
    if not data:
        raise RuntimeError(f"No models found at {base_url}")
    return data[0]["id"]


def run_streaming(base_url: str, model: str, messages: list, max_tokens: int = 512,
                  temperature: float = 0.0, tools: list = None,
                  tool_choice: str = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Send a streaming chat completion request. Returns dict with:
      completion, tool_calls, ttft_ms, total_latency_ms, tokens_generated, tokens_per_sec
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    t_start = time.perf_counter()
    t_first_token = None
    chunks_text = []
    tool_calls_acc = {}  # index -> {id, name, arguments}
    usage = {}

    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Extract usage from final chunk
        if chunk.get("usage"):
            usage = chunk["usage"]

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})

        # Text content
        content = delta.get("content")
        if content:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            chunks_text.append(content)

        # Tool calls
        tc_list = delta.get("tool_calls")
        if tc_list:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            for tc in tc_list:
                idx = tc.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                if tc.get("id"):
                    tool_calls_acc[idx]["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    tool_calls_acc[idx]["name"] = fn["name"]
                if fn.get("arguments"):
                    tool_calls_acc[idx]["arguments"] += fn["arguments"]

    t_end = time.perf_counter()

    completion = "".join(chunks_text)
    total_latency = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_latency

    # Token counts from usage or estimate
    tokens_generated = usage.get("completion_tokens", 0)
    if tokens_generated == 0:
        tokens_generated = max(len(completion.split()), 1)

    generation_time = (t_end - t_first_token) if t_first_token else total_latency
    tps = tokens_generated / generation_time if generation_time > 0 else 0

    # Parse tool calls
    parsed_tool_calls = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        try:
            args = json.loads(tc["arguments"]) if tc["arguments"] else {}
        except json.JSONDecodeError:
            args = tc["arguments"]
        parsed_tool_calls.append({
            "id": tc["id"],
            "name": tc["name"],
            "arguments": args,
        })

    return {
        "completion": completion,
        "tool_calls": parsed_tool_calls,
        "ttft_ms": round(ttft * 1000, 1),
        "total_latency_ms": round(total_latency * 1000, 1),
        "tokens_generated": tokens_generated,
        "tokens_per_sec": round(tps, 1),
        "prompt_tokens": usage.get("prompt_tokens", 0),
    }


def run_non_streaming(base_url: str, model: str, messages: list, max_tokens: int = 512,
                      temperature: float = 0.0, tools: list = None,
                      tool_choice: str = None, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fallback non-streaming call (used for multi-turn tool tests)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    t_start = time.perf_counter()
    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    t_end = time.perf_counter()

    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})

    completion = msg.get("content", "") or ""
    tool_calls = []
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = fn.get("arguments", "")
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            })

    tokens_generated = usage.get("completion_tokens", max(len(completion.split()), 1))
    total = t_end - t_start
    tps = tokens_generated / total if total > 0 else 0

    return {
        "completion": completion,
        "tool_calls": tool_calls,
        "ttft_ms": round(total * 1000, 1),
        "total_latency_ms": round(total * 1000, 1),
        "tokens_generated": tokens_generated,
        "tokens_per_sec": round(tps, 1),
        "prompt_tokens": usage.get("prompt_tokens", 0),
    }


def compute_similarity(text_a: str, text_b: str) -> float:
    """Word-level overlap similarity."""
    a = text_a.split()
    b = text_b.split()
    if not a or not b:
        return 1.0 if a == b else 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / max(len(a), len(b))


def extract_number(text: str):
    """Extract the last number from text for numerical answer checking."""
    # Look for boxed answers first
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        try:
            return float(boxed[-1].replace(",", "").replace("$", ""))
        except ValueError:
            pass
    # Fall back to last number in text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[-1])
    return None


# ---------------------------------------------------------------------------
# Seed text for long-context tests
# ---------------------------------------------------------------------------

SEED_TEXT = (
    "The development of transformer architectures revolutionized natural language "
    "processing. Starting with the original attention mechanism proposed in 2017, "
    "researchers discovered that self-attention could capture long-range dependencies "
    "far more effectively than recurrent networks. The key innovation was the ability "
    "to process all positions in parallel, which not only improved training speed but "
    "also allowed models to learn richer representations of language. As models grew "
    "larger, from millions to billions of parameters, they began exhibiting emergent "
    "capabilities that surprised even their creators. In-context learning, chain of "
    "thought reasoning, and few-shot generalization appeared without explicit training. "
    "Quantum error correction is essential for building practical quantum computers "
    "because qubits are inherently fragile. Unlike classical bits that are either 0 "
    "or 1, qubits exist in superposition states that can be disrupted by environmental "
    "interactions. The surface code arranges physical qubits in a two-dimensional grid. "
)

NEEDLE_FACT = "The secret password for project Orion is 'blue-harvest-7742'."


def build_haystack_prompt(target_chars: int) -> str:
    """Build a long prompt with a needle fact buried in the middle."""
    # Build filler to target size
    filler = ""
    while len(filler) < target_chars:
        filler += SEED_TEXT + " "

    # Insert needle at ~40% through the text
    insert_pos = int(len(filler) * 0.4)
    filler = filler[:insert_pos] + f"\n\n{NEEDLE_FACT}\n\n" + filler[insert_pos:]
    filler = filler[:target_chars]

    return filler + "\n\nWhat is the secret password for project Orion?"


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
}

RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read a webpage and return its content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to read"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize a piece of text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                },
                "required": ["text"],
            },
        },
    },
]

FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a pattern in the codebase",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex)"},
                    "path": {"type": "string", "description": "Directory to search in"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def build_test_suite(categories: list[str] | None = None) -> list[dict]:
    """Build the full test suite. Each test is a dict with category, name, messages, etc."""
    tests = []

    # -----------------------------------------------------------------------
    # Code Generation
    # -----------------------------------------------------------------------
    if categories is None or "code" in categories:
        tests.append({
            "category": "code",
            "name": "merge_sorted_lists",
            "messages": [
                {"role": "user", "content":
                 "Write a Python function `merge_sorted_lists(a, b)` that merges two sorted "
                 "lists into one sorted list in O(n+m) time. Do not use .sort() or sorted(). "
                 "Return only the function, no explanation."}
            ],
            "max_tokens": 300,
            "validator": lambda r: (
                "def merge_sorted_lists" in r["completion"]
                and ".sort()" not in r["completion"]
                and "sorted(" not in r["completion"]
            ),
        })

        tests.append({
            "category": "code",
            "name": "debug_binary_search",
            "messages": [
                {"role": "user", "content":
                 "This binary search function has a bug. Find and fix it:\n\n"
                 "```python\n"
                 "def binary_search(arr, target):\n"
                 "    lo, hi = 0, len(arr)\n"
                 "    while lo < hi:\n"
                 "        mid = (lo + hi) // 2\n"
                 "        if arr[mid] == target:\n"
                 "            return mid\n"
                 "        elif arr[mid] < target:\n"
                 "            lo = mid\n"
                 "        else:\n"
                 "            hi = mid\n"
                 "    return -1\n"
                 "```\n\n"
                 "Explain the bug and provide the corrected code."}
            ],
            "max_tokens": 500,
            "validator": lambda r: (
                "lo = mid + 1" in r["completion"] or "lo = mid+1" in r["completion"]
            ),
        })

        tests.append({
            "category": "code",
            "name": "explain_regex",
            "messages": [
                {"role": "user", "content":
                 "Explain what this regex does step by step:\n\n"
                 r"`^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$`"}
            ],
            "max_tokens": 400,
            "validator": lambda r: (
                "ip" in r["completion"].lower() or "address" in r["completion"].lower()
            ),
        })

        tests.append({
            "category": "code",
            "name": "palindrome_dp",
            "messages": [
                {"role": "user", "content":
                 "Write a Python function `longest_palindrome(s)` that finds the longest palindromic "
                 "substring using dynamic programming (2D table approach, not expand-around-center). "
                 "Return only the function."}
            ],
            "max_tokens": 500,
            "validator": lambda r: (
                "def longest_palindrome" in r["completion"]
                and ("dp" in r["completion"].lower() or "table" in r["completion"].lower()
                     or "[[" in r["completion"])
            ),
        })

    # -----------------------------------------------------------------------
    # Reasoning
    # -----------------------------------------------------------------------
    if categories is None or "reasoning" in categories:
        tests.append({
            "category": "reasoning",
            "name": "arithmetic",
            "messages": [
                {"role": "user", "content":
                 "A store sells notebooks for $4 each and pens for $2 each. Sarah buys 3 notebooks "
                 "and 5 pens. She pays with a $30 bill. How much change does she get? "
                 "Show your work and give the final answer as a number."}
            ],
            "max_tokens": 300,
            "expected_answer": 8.0,
            "validator": lambda r: extract_number(r["completion"]) == 8.0,
        })

        tests.append({
            "category": "reasoning",
            "name": "work_rate",
            "messages": [
                {"role": "user", "content":
                 "Alice can paint a fence in 6 hours. Bob can paint the same fence in 3 hours. "
                 "If they work together, how many hours will it take them to paint the fence? "
                 "Give the final answer as a number."}
            ],
            "max_tokens": 300,
            "expected_answer": 2.0,
            "validator": lambda r: extract_number(r["completion"]) == 2.0,
        })

        tests.append({
            "category": "reasoning",
            "name": "combinatorics",
            "messages": [
                {"role": "user", "content":
                 "How many ways can you choose 3 items from a set of 7? "
                 "Give the final answer as a number."}
            ],
            "max_tokens": 300,
            "expected_answer": 35.0,
            "validator": lambda r: extract_number(r["completion"]) == 35.0,
        })

        tests.append({
            "category": "reasoning",
            "name": "logic_puzzle",
            "messages": [
                {"role": "user", "content":
                 "Three friends (Alice, Bob, Charlie) sit in chairs 1, 2, 3 (left to right). "
                 "Clues: Alice is not in chair 1. Bob is not in chair 3. Charlie is not in chair 2. "
                 "What is the seating order from chair 1 to 3? "
                 "Give only the names separated by commas."}
            ],
            "max_tokens": 300,
            "expected_answer": "Bob, Alice, Charlie",
            "validator": lambda r: (
                "bob" in r["completion"].lower()
                and "alice" in r["completion"].lower()
                and "charlie" in r["completion"].lower()
                and r["completion"].lower().index("bob") < r["completion"].lower().index("alice")
                < r["completion"].lower().index("charlie")
            ),
        })

    # -----------------------------------------------------------------------
    # Scaffolding / Tool-use
    # -----------------------------------------------------------------------
    if categories is None or "scaffold" in categories:
        tests.append({
            "category": "scaffold",
            "name": "single_tool_call",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
                {"role": "user", "content": "What's the weather like in Tokyo in celsius?"},
            ],
            "max_tokens": 200,
            "tools": [WEATHER_TOOL],
            "validator": lambda r: (
                len(r["tool_calls"]) > 0
                and r["tool_calls"][0]["name"] == "get_weather"
                and "tokyo" in json.dumps(r["tool_calls"][0]["arguments"]).lower()
            ),
        })

        tests.append({
            "category": "scaffold",
            "name": "multi_tool_planning",
            "messages": [
                {"role": "system", "content":
                 "You are a research assistant. You have access to web search, page reading, and "
                 "summarization tools. Use them to answer questions step by step."},
                {"role": "user", "content":
                 "Find and compare the population of Iceland and Greenland."},
            ],
            "max_tokens": 300,
            "tools": RESEARCH_TOOLS,
            "validator": lambda r: (
                len(r["tool_calls"]) > 0
                and r["tool_calls"][0]["name"] == "search_web"
            ),
        })

        tests.append({
            "category": "scaffold",
            "name": "structured_json_output",
            "messages": [
                {"role": "user", "content":
                 "Return a JSON array of 3 items sold in a grocery store. Each item should have "
                 "fields: name (string), price (number), category (string). "
                 "Return ONLY valid JSON, nothing else."}
            ],
            "max_tokens": 300,
            "validator": lambda r: _validate_json_array(r["completion"]),
        })

        # Multi-turn: user asks → model calls tool → we inject result → model answers
        tests.append({
            "category": "scaffold",
            "name": "multi_turn_tool_use",
            "multi_turn": True,
            "messages_phase1": [
                {"role": "system", "content":
                 "You are a coding assistant. Use the provided tools to help the user."},
                {"role": "user", "content":
                 "Read the file /app/config.json and tell me what port the server runs on."},
            ],
            "tool_result": json.dumps({"host": "0.0.0.0", "port": 8080, "debug": False}),
            "max_tokens": 200,
            "tools": FILE_TOOLS,
            "validator": lambda r: "8080" in r["completion"],
        })

    # -----------------------------------------------------------------------
    # General Chat
    # -----------------------------------------------------------------------
    if categories is None or "chat" in categories:
        tests.append({
            "category": "chat",
            "name": "summarization",
            "messages": [
                {"role": "user", "content":
                 "Summarize the following in 2-3 sentences:\n\n"
                 "The transformer architecture, introduced in 2017, replaced recurrent neural "
                 "networks for most NLP tasks by using self-attention to process all input positions "
                 "in parallel. This dramatically improved training speed and enabled models to "
                 "capture long-range dependencies more effectively. The architecture consists of "
                 "an encoder-decoder structure, though many modern variants use only the decoder "
                 "(like GPT) or only the encoder (like BERT). Scaling these models to billions of "
                 "parameters revealed emergent capabilities such as in-context learning, chain-of-thought "
                 "reasoning, and few-shot generalization. These capabilities were not explicitly trained "
                 "but appeared as a byproduct of scale, fundamentally changing how we approach AI."}
            ],
            "max_tokens": 200,
            "validator": lambda r: (
                "transformer" in r["completion"].lower()
                and len(r["completion"].split(".")) >= 2
                and len(r["completion"].split()) < 150
            ),
        })

        tests.append({
            "category": "chat",
            "name": "factual_qa",
            "messages": [
                {"role": "user", "content":
                 "Answer these three questions concisely:\n"
                 "1. What is the chemical symbol for gold?\n"
                 "2. Who wrote 'Romeo and Juliet'?\n"
                 "3. What is the speed of light in km/s (approximately)?"}
            ],
            "max_tokens": 200,
            "validator": lambda r: (
                "au" in r["completion"].lower()
                and "shakespeare" in r["completion"].lower()
                and ("300" in r["completion"] or "299" in r["completion"])
            ),
        })

        tests.append({
            "category": "chat",
            "name": "instruction_following",
            "messages": [
                {"role": "user", "content":
                 "Follow these instructions exactly:\n"
                 "1. Write the word 'BEGIN'\n"
                 "2. List exactly 5 programming languages, one per line, numbered 1-5\n"
                 "3. Write the word 'END'\n"
                 "Do not add any other text."}
            ],
            "max_tokens": 200,
            "validator": lambda r: (
                "BEGIN" in r["completion"]
                and "END" in r["completion"]
                and any(str(i) in r["completion"] for i in range(1, 6))
            ),
        })

        tests.append({
            "category": "chat",
            "name": "creative_writing",
            "messages": [
                {"role": "user", "content":
                 "Write a haiku (5-7-5 syllable poem) about a computer. "
                 "Return only the three lines of the haiku, nothing else."}
            ],
            "max_tokens": 100,
            "validator": lambda r: (
                len([line for line in r["completion"].strip().split("\n") if line.strip()]) >= 3
            ),
        })

        tests.append({
            "category": "chat",
            "name": "multi_turn_conversation",
            "messages": [
                {"role": "user", "content": "My name is Alex. I'm learning about quantum computing."},
                {"role": "assistant", "content":
                 "Nice to meet you, Alex! Quantum computing is a fascinating field. "
                 "What aspect are you most interested in?"},
                {"role": "user", "content":
                 "I'm curious about qubits. Can you explain what they are in simple terms? "
                 "Also, do you remember my name?"},
            ],
            "max_tokens": 400,
            "validator": lambda r: (
                "alex" in r["completion"].lower()
                and ("qubit" in r["completion"].lower() or "quantum" in r["completion"].lower())
            ),
        })

    # -----------------------------------------------------------------------
    # Long Context
    # -----------------------------------------------------------------------
    if "longctx" in (categories or []):
        for target_k, chars in [("8K", 32_000), ("32K", 128_000), ("64K", 256_000), ("90K", 360_000)]:
            tests.append({
                "category": "longctx",
                "name": f"needle_{target_k}",
                "messages": [
                    {"role": "user", "content": build_haystack_prompt(chars)},
                ],
                "max_tokens": 100,
                "timeout": LONG_CTX_TIMEOUT,
                "validator": lambda r: "blue-harvest-7742" in r["completion"].lower(),
            })

    return tests


def _validate_json_array(text: str) -> bool:
    """Check that text contains a valid JSON array with required fields."""
    # Try to extract JSON from possible markdown code blocks
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, list) or len(data) < 3:
            return False
        for item in data[:3]:
            if not all(k in item for k in ("name", "price", "category")):
                return False
        return True
    except (json.JSONDecodeError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Multi-turn tool-use handler
# ---------------------------------------------------------------------------

def run_multi_turn_test(base_url: str, model: str, test: dict, timeout: int) -> dict:
    """
    Phase 1: Send initial messages, expect a tool call.
    Phase 2: Inject tool result, get final answer.
    Returns combined metrics.
    """
    # Phase 1: get the tool call
    result1 = run_streaming(
        base_url, model, test["messages_phase1"],
        max_tokens=test["max_tokens"], tools=test.get("tools"),
        timeout=timeout,
    )

    if not result1["tool_calls"]:
        # Model didn't call a tool — return as-is
        return result1

    # Phase 2: inject tool result and get final answer
    tc = result1["tool_calls"][0]
    messages = list(test["messages_phase1"]) + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc["id"] or "call_001",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": tc["id"] or "call_001",
            "content": test["tool_result"],
        },
    ]

    result2 = run_streaming(
        base_url, model, messages,
        max_tokens=test["max_tokens"], tools=test.get("tools"),
        timeout=timeout,
    )

    # Combine metrics
    return {
        "completion": result2["completion"],
        "tool_calls": result1["tool_calls"],
        "ttft_ms": result1["ttft_ms"],  # first interaction TTFT
        "total_latency_ms": result1["total_latency_ms"] + result2["total_latency_ms"],
        "tokens_generated": result1["tokens_generated"] + result2["tokens_generated"],
        "tokens_per_sec": result2["tokens_per_sec"],  # use phase 2 TPS
        "prompt_tokens": result1["prompt_tokens"] + result2["prompt_tokens"],
        "phase1_tool_call": result1["tool_calls"][0] if result1["tool_calls"] else None,
        "phase2_completion": result2["completion"],
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(endpoints: dict, tests: list, verbose: bool = False) -> dict:
    """Run all tests against all endpoints, collect metrics."""
    # Detect models
    models = {}
    for key, ep in endpoints.items():
        try:
            models[key] = detect_model(ep["base_url"])
            print(f"  {ep['label']}: {models[key]}")
        except Exception as e:
            print(f"  {ep['label']}: UNREACHABLE ({e})")
            models[key] = None

    print()

    results = []
    total = len(tests)
    for i, test in enumerate(tests, 1):
        name = f"{test['category']}: {test['name']}"
        print(f"[{i:2d}/{total}] {name:40s}", end="", flush=True)

        test_result = {"category": test["category"], "name": test["name"]}
        timeout = test.get("timeout", DEFAULT_TIMEOUT)

        for key, ep in endpoints.items():
            if models[key] is None:
                test_result[key] = {"error": "endpoint unreachable"}
                continue

            try:
                if test.get("multi_turn"):
                    result = run_multi_turn_test(ep["base_url"], models[key], test, timeout)
                else:
                    result = run_streaming(
                        ep["base_url"], models[key], test["messages"],
                        max_tokens=test["max_tokens"],
                        tools=test.get("tools"),
                        timeout=timeout,
                    )

                # Run validator
                passed = False
                try:
                    if test.get("validator"):
                        passed = test["validator"](result)
                except Exception:
                    passed = False

                test_result[key] = {
                    "completion": result["completion"][:2000],
                    "tool_calls": result["tool_calls"],
                    "ttft_ms": result["ttft_ms"],
                    "total_latency_ms": result["total_latency_ms"],
                    "tokens_generated": result["tokens_generated"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "passed": passed,
                }

            except requests.exceptions.Timeout:
                test_result[key] = {"error": "timeout", "passed": False}
            except Exception as e:
                test_result[key] = {"error": str(e)[:200], "passed": False}

        # Compute comparison (only if 2+ endpoints)
        if len(endpoints) >= 2:
            keys = list(endpoints.keys())
            r1 = test_result.get(keys[0], {})
            r2 = test_result.get(keys[1], {})
            if "error" not in r1 and "error" not in r2:
                test_result["comparison"] = {
                    "similarity": round(compute_similarity(
                        r1.get("completion", ""), r2.get("completion", "")
                    ), 3),
                    "both_passed": r1.get("passed", False) and r2.get("passed", False),
                    "ttft_ratio": round(r1["ttft_ms"] / r2["ttft_ms"], 2) if r2.get("ttft_ms") else None,
                    "tps_ratio": round(r1["tokens_per_sec"] / r2["tokens_per_sec"], 2) if r2.get("tokens_per_sec") else None,
                }

        # Print inline status
        status_parts = []
        for key, ep in endpoints.items():
            r = test_result.get(key, {})
            s = "Y" if r.get("passed") else ("E" if "error" in r else "N")
            tps = f"{r.get('tokens_per_sec', 0):>6.1f}" if "error" not in r else "  err "
            status_parts.append(f"{ep['label'][:12]}[{s}]{tps}t/s")
        print(f"  {'  '.join(status_parts)}")

        if verbose:
            for key, ep in endpoints.items():
                r = test_result.get(key, {})
                if "error" not in r:
                    print(f"    {ep['label']}: {r.get('completion', '')[:120]}")

        results.append(test_result)

    return {"tests": results, "models": models}


def print_summary(results: dict, endpoints: dict):
    """Print the summary table."""
    tests = results["tests"]
    ep_keys = list(endpoints.keys())

    print(f"\n{'='*90}")
    print(f"MODEL EVALUATION SUMMARY")
    print(f"{'='*90}")

    # Header
    header = f"{'Test':<35}"
    subhdr = f"{'':35}"
    for key in ep_keys:
        label = endpoints[key]["label"][:20]
        header += f" | {label:^30}"
        subhdr += f" | {'TTFT':>7} {'TPS':>7} {'Tok':>5} {'Pass':>4}"
    print(header)
    print(subhdr)
    print("-" * 90)

    cat_stats = {key: {} for key in ep_keys}
    for t in tests:
        cat = t["category"]
        name = f"{cat}: {t['name']}"
        if len(name) > 34:
            name = name[:31] + "..."

        row = f"{name:<35}"
        for key in ep_keys:
            r = t.get(key, {})
            if "error" in r:
                row += f" | {'err':>7} {'':>7} {'':>5} {'':>4}"
            else:
                row += f" | {r['ttft_ms']:>6.0f}m {r['tokens_per_sec']:>6.1f} {r['tokens_generated']:>5} {'Y' if r.get('passed') else 'N':>4}"

            # Aggregate stats
            if cat not in cat_stats[key]:
                cat_stats[key][cat] = []
            if "error" not in r:
                cat_stats[key][cat].append(r)

        print(row)

    print("-" * 90)

    # Category summaries
    all_cats = sorted(set(cat for key in ep_keys for cat in cat_stats[key]))
    for cat in all_cats:
        row = f"  {cat:<33}"
        for key in ep_keys:
            data = cat_stats[key].get(cat, [])
            if data:
                avg_tps = sum(x["tokens_per_sec"] for x in data) / len(data)
                avg_ttft = sum(x["ttft_ms"] for x in data) / len(data)
                pass_rate = sum(1 for x in data if x.get("passed")) / len(data) * 100
                row += f" | {avg_ttft:>6.0f}m {avg_tps:>6.1f} {'':>5} {pass_rate:>3.0f}%"
            else:
                row += f" | {'':>30}"
        print(row)

    # Grand totals
    print(f"\n{'='*90}")
    for key in ep_keys:
        all_results = [t.get(key, {}) for t in tests if "error" not in t.get(key, {"error": 1})]
        if all_results:
            avg_tps = sum(x["tokens_per_sec"] for x in all_results) / len(all_results)
            avg_ttft = sum(x["ttft_ms"] for x in all_results) / len(all_results)
            pass_rate = sum(1 for x in all_results if x.get("passed")) / len(all_results) * 100
            print(f"  {endpoints[key]['label']}  avg: TTFT={avg_ttft:.0f}ms  TPS={avg_tps:.1f}  Pass={pass_rate:.0f}%  ({len(all_results)} tests)")

    if len(ep_keys) >= 2:
        sims = [t["comparison"]["similarity"] for t in tests if "comparison" in t]
        if sims:
            print(f"  Avg output similarity: {sum(sims)/len(sims):.3f}")

    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation: Qwen3-Coder-Next-AWQ-4bit")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories to run: code,reasoning,scaffold,chat,longctx")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: ~/eval_results_TIMESTAMP.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show output snippets during run")
    args = parser.parse_args()

    categories = args.categories.split(",") if args.categories else None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(Path.home() / f"eval_results_{timestamp}.json")

    print(f"{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Timestamp: {timestamp}")
    print(f"Categories: {categories or 'all'}")
    print(f"Output: {output_path}")
    print(f"\nDetecting models...")

    tests = build_test_suite(categories)
    print(f"Tests to run: {len(tests)}\n")

    results = run_evaluation(ENDPOINTS, tests, verbose=args.verbose)

    # Add metadata
    results["metadata"] = {
        "timestamp": timestamp,
        "endpoints": {k: v["label"] for k, v in ENDPOINTS.items()},
        "categories": categories or ["code", "reasoning", "scaffold", "chat"],
        "total_tests": len(tests),
    }

    # Save JSON
    # Strip validators (not serializable) from results — they're already evaluated
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print_summary(results, ENDPOINTS)


if __name__ == "__main__":
    main()
