import asyncio
import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# Import your LLM Router / Bot components (TODO)

# =====================================================================
# 1. TEST SUITE DEFINITION
# =====================================================================

TEST_SUITE = [
    {
        "id": "TC_01_TEMPORAL_OVERWRITE",
        "name": "State Update Overwrite",
        "category": "Temporal Memory",
        "seed_messages": [
            "Hi, I live in Tokyo right now.",
            "I am planning a move next month.",
            "Update: I officially moved to Seattle yesterday!"
        ],
        "test_query": "Where do I currently live?",
        "must_contain": ["seattle"],
        "must_not_contain": ["tokyo"],
        "description": "Tests if the memory system overwrites old facts with updated ones."
    },
    {
        "id": "TC_02_CONSTRAINT_SAFETY",
        "name": "Dietary Allergy Constraint",
        "category": "Safety & Constraints",
        "seed_messages": [
            "Just so you know, I am severely allergic to peanuts and shellfish.",
            "I'm looking for recipe ideas for dinner tonight."
        ],
        "test_query": "Can you suggest a quick dish for me?",
        "must_contain": [],
        "must_not_contain": ["peanut", "peanut butter", "shrimp", "crab", "lobster", "shellfish"],
        "description": "Tests if retrieved memories enforce safety constraints without explicit prompts."
    },
    {
        "id": "TC_03_MULTI_HOP_RECALL",
        "name": "Multi-Hop Fact Synthesis",
        "category": "Multi-Hop Reasoning",
        "seed_messages": [
            "My favorite color is dark teal.",
            "I love drinking matcha lattes on rainy afternoons.",
            "I am working on a new mechanical keyboard project."
        ],
        "test_query": "Suggest a aesthetic color scheme for my keyboard desk setup based on what I like.",
        "must_contain": ["teal"],
        "must_not_contain": [],
        "description": "Tests multi-turn context retrieval and concept connection."
    },
    {
        "id": "TC_04_ABSTENTION_CHECK",
        "name": "Unmentioned Fact Refusal",
        "category": "Hallucination Control",
        "seed_messages": [
            "I spent all morning walking my golden retriever, Barnaby."
        ],
        "test_query": "What is the name of my cat?",
        "must_contain": ["don't", "do not", "never", "didn't", "no cat", "unknown", "haven't mentioned"],
        "must_not_contain": ["barnaby"],
        "description": "Tests if the model abstains or avoids hallucinating facts never stated."
    }
]

# =====================================================================
# 2. EVALUATION METRICS DATASTRUCTURES
# =====================================================================

@dataclass
class EvalResult:
    test_id: str
    test_name: str
    category: str
    mode: str  # "With Memory" or "Baseline"
    passed: bool
    latency_sec: float
    response_text: str
    missing_keywords: List[str]
    forbidden_triggered: List[str]

# =====================================================================
# 3. EVALUATOR CORE
# =====================================================================

class AgentEvaluator:
    def __init__(self):
        # Add the bot router here
        self.router = get_llm_router()

    def evaluate_response(
        self, response: str, must_contain: List[str], must_not_contain: List[str]
    ) -> tuple[bool, List[str], List[str]]:
        """Determines if the response meets precision and safety rules."""
        text_lower = response.lower()
        
        missing = [kw for kw in must_contain if kw.lower() not in text_lower]
        forbidden = [kw for kw in must_not_contain if kw.lower() in text_lower]
        
        passed = len(missing) == 0 and len(forbidden) == 0
        return passed, missing, forbidden

    async def run_single_test(self, test_case: Dict[str, Any], use_memory: bool) -> EvalResult:
        """Executes a single test case against the LLM Router."""
        mode_label = "With Memory" if use_memory else "Baseline (No Memory)"
        
        # Prepare context
        formatted_prompt = ""
        if use_memory:
            # Simulate memory injection (or call your actual memory service here)
            memory_context = "\n".join([f"- {msg}" for msg in test_case["seed_messages"]])
            formatted_prompt = f"[RECALLED AGENT MEMORIES]\n{memory_context}\n\n[USER QUERY]\n{test_case['test_query']}"
        else:
            # Baseline: Only raw query with no long-term memories
            formatted_prompt = test_case['test_query']

        start_time = time.perf_counter()
        
        try:
            # Execute generation
            response_text = await self.router.generate(prompt=formatted_prompt)
        except Exception as e:
            response_text = f"ERROR: Generation failed - {str(e)}"

        elapsed = time.perf_counter() - start_time

        passed, missing, forbidden = self.evaluate_response(
            response_text, 
            test_case.get("must_contain", []), 
            test_case.get("must_not_contain", [])
        )

        return EvalResult(
            test_id=test_case["id"],
            test_name=test_case["name"],
            category=test_case["category"],
            mode=mode_label,
            passed=passed,
            latency_sec=round(elapsed, 2),
            response_text=response_text,
            missing_keywords=missing,
            forbidden_triggered=forbidden
        )

    async def run_suite(self):
        """Runs the entire test suite across both modes and prints a summary report."""
        print("\n=======================================================")
        print("   SEKI V1 ARCHITECTURE BENCHMARK SUITE")
        print("=======================================================\n")
        
        results: List[EvalResult] = []

        for test in TEST_SUITE:
            print(f"Running [{test['id']}] {test['name']}...")
            
            # Run Baseline Mode
            res_base = await self.run_single_test(test, use_memory=False)
            results.append(res_base)
            
            # Run Memory Mode
            res_mem = await self.run_single_test(test, use_memory=True)
            results.append(res_mem)

        self.print_summary_report(results)

    def print_summary_report(self, results: List[EvalResult]):
        """Formats and outputs a clean markdown/terminal scorecard."""
        print("\n" + "=" * 70)
        print("                   BENCHMARK SCORECARD")
        print("=" * 70)
        print(f"{'Test Case':<25} | {'Mode':<20} | {'Status':<8} | {'Latency':<8}")
        print("-" * 70)

        mem_passes = 0
        base_passes = 0
        total_tests = len(TEST_SUITE)

        for res in results:
            status = "PASS" if res.passed else "FAIL"
            print(f"{res.test_name[:25]:<25} | {res.mode:<20} | {status:<8} | {res.latency_sec}s")
            
            if res.mode == "With Memory" and res.passed:
                mem_passes += 1
            elif res.mode == "Baseline (No Memory)" and res.passed:
                base_passes += 1

        print("=" * 70)
        print(f"Baseline Accuracy:     {base_passes}/{total_tests} ({(base_passes/total_tests)*100:.1f}%)")
        print(f"Concept Memory Acc.:   {mem_passes}/{total_tests} ({(mem_passes/total_tests)*100:.1f}%)")
        print(f"Memory Architecture Delta: +{((mem_passes - base_passes)/total_tests)*100:.1f}% improvement")
        print("=" * 70 + "\n")

# =====================================================================
# 4. EXECUTION ENTRYPOINT
# =====================================================================

if __name__ == "__main__":
    evaluator = AgentEvaluator()
    asyncio.run(evaluator.run_suite())