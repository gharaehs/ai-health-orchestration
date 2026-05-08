"""
rag/pipeline.py

Main RAG orchestration class for the AI Health Orchestration System.

Ties together:
  1. HealthRetriever  — queries ChromaDB with the right collection routing
  2. HealthPromptBuilder — assembles the augmented prompt
  3. vLLM API call    — generates the structured JSON health plan

Usage:
    pipeline = HealthRAGPipeline()
    result   = pipeline.query(
        health_profile={"age": 32, "weight_kg": 88, ...},
        goal="fat_loss",
        output_type="meal_plan",
    )
    print(result.response_json)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from rag.retriever import HealthRetriever, RetrievedChunk
from rag.prompt_builder import HealthPromptBuilder, PromptStrategy

logger = logging.getLogger(__name__)


# ── Output type → query routing ───────────────────────────────────────────────
# Maps the requested output type to:
#   - the collection(s) to retrieve from
#   - the number of chunks to pull per collection
#   - the natural language query template for retrieval

OUTPUT_TYPE_CONFIG = {
    "meal_plan": {
        "query_type":              "meal_plan",
        "n_results_per_collection": 5,
        "query_template": (
            "meal plan nutrition guidelines macros calories protein carbohydrates fat "
            "{goal} {dietary_restrictions}"
        ),
    },
    "gym_program": {
        "query_type":              "gym_program",
        "n_results_per_collection": 6,
        "query_template": (
            "resistance training program sets reps progression {goal} "
            "periodization hypertrophy strength"
        ),
    },
    "grocery_list": {
        "query_type":              "grocery",
        "n_results_per_collection": 5,
        "query_template": (
            "healthy food ingredients recipes {goal} {dietary_restrictions} "
            "grocery nutrition"
        ),
    },
    "lab_analysis": {
        "query_type":              "lab_analysis",
        "n_results_per_collection": 6,
        "query_template": (
            "blood test reference ranges lab values {markers} interpretation "
            "clinical thresholds"
        ),
    },
    "full_plan": {
        "query_type":              "full",
        "n_results_per_collection": 3,
        "query_template": (
            "health plan nutrition exercise meal program {goal} "
            "{dietary_restrictions} blood markers"
        ),
    },
}

# ── JSON output schemas ───────────────────────────────────────────────────────
# Injected into the user message so the model knows the exact output format.

OUTPUT_SCHEMAS = {
    "meal_plan": """{
  "meal_plan": {
    "daily_calories": <number>,
    "macros": {"protein_g": <number>, "carbs_g": <number>, "fat_g": <number>},
    "week": [
      {
        "day": "<Monday|Tuesday|...>",
        "meals": {
          "breakfast": {"name": "<str>", "calories": <number>, "protein_g": <number>, "carbs_g": <number>, "fat_g": <number>},
          "lunch":     {"name": "<str>", "calories": <number>, "protein_g": <number>, "carbs_g": <number>, "fat_g": <number>},
          "dinner":    {"name": "<str>", "calories": <number>, "protein_g": <number>, "carbs_g": <number>, "fat_g": <number>},
          "snacks":    [{"name": "<str>", "calories": <number>}]
        }
      }
    ]
  }
}""",
    "gym_program": """{
  "gym_program": {
    "goal": "<str>",
    "frequency_days_per_week": <number>,
    "weeks": [
      {
        "week": <number>,
        "sessions": [
          {
            "day": "<str>",
            "focus": "<str>",
            "exercises": [
              {
                "name": "<str>",
                "sets": <number>,
                "reps": "<str>",
                "rest_seconds": <number>,
                "notes": "<str>"
              }
            ]
          }
        ]
      }
    ]
  }
}""",
    "grocery_list": """{
  "grocery_list": {
    "categories": {
      "proteins":     [{"item": "<str>", "quantity": "<str>", "notes": "<str>"}],
      "vegetables":   [{"item": "<str>", "quantity": "<str>"}],
      "fruits":       [{"item": "<str>", "quantity": "<str>"}],
      "grains":       [{"item": "<str>", "quantity": "<str>"}],
      "dairy":        [{"item": "<str>", "quantity": "<str>"}],
      "fats_oils":    [{"item": "<str>", "quantity": "<str>"}],
      "pantry":       [{"item": "<str>", "quantity": "<str>"}]
    },
    "estimated_weekly_cost_eur": <number>
  }
}""",
    "lab_analysis": """{
  "lab_analysis": {
    "summary": "<str>",
    "markers": [
      {
        "name": "<str>",
        "value": "<str>",
        "unit": "<str>",
        "reference_range": "<str>",
        "status": "<normal|borderline|elevated|low>",
        "interpretation": "<str>",
        "recommendation": "<str>"
      }
    ],
    "overall_risk_assessment": "<str>",
    "follow_up_recommendations": ["<str>"]
  }
}""",
    "full_plan": """{
  "plan": {
    "meal_plan": { ... },
    "gym_program": { ... },
    "grocery_list": { ... }
  }
}""",
}


@dataclass
class RAGResult:
    """Complete result object from a RAG pipeline query."""
    output_type:       str
    query_type_used:   str
    strategy_used:     str
    n_chunks_retrieved: int
    chunks:            list[RetrievedChunk]
    prompt_token_est:  int
    response_raw:      str
    response_json:     Optional[dict]
    parse_error:       Optional[str]
    latency_seconds:   float
    rag_enabled:       bool


class HealthRAGPipeline:
    """
    End-to-end RAG pipeline for the AI Health Orchestration System.

    Retrieves relevant health science context from ChromaDB, assembles
    an augmented prompt, and generates a structured JSON health plan via
    the local vLLM instance.
    """

    def __init__(
        self,
        vllm_url:       str = "http://localhost:8000",
        chroma_host:    str = "localhost",
        chroma_port:    int = 8001,
        model_name:     str = "/models/llama",
        max_tokens:     int = 2048,
        temperature:    float = 0.3,      # Low temp for structured JSON output
        strategy:       PromptStrategy = PromptStrategy.STRUCTURED,
        max_context_chars: int = 12_000,
    ):
        self.vllm_url   = vllm_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.strategy   = strategy

        self.retriever = HealthRetriever(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
        )
        self.prompt_builder = HealthPromptBuilder(
            max_context_chars=max_context_chars,
        )

        logger.info(
            f"HealthRAGPipeline ready — model: {model_name}, "
            f"strategy: {strategy.value}, max_tokens: {max_tokens}"
        )

    def query(
        self,
        health_profile: dict,
        goal: str,
        output_type: str = "meal_plan",
        extra_context: Optional[str] = None,
        rag_enabled: bool = True,
        override_strategy: Optional[PromptStrategy] = None,
    ) -> RAGResult:
        """
        Run the full RAG pipeline for a health query.

        Args:
            health_profile: Dict with user health data (age, weight, blood markers, etc.)
            goal:           Health goal string (e.g. "fat_loss", "muscle_gain")
            output_type:    One of: meal_plan, gym_program, grocery_list,
                            lab_analysis, full_plan
            extra_context:  Optional free-text additional instructions
            rag_enabled:    Set False to run without retrieval (baseline comparison)
            override_strategy: Override the default prompt strategy for this call

        Returns:
            RAGResult with response_json (parsed) and all pipeline metadata
        """
        t_start = time.time()

        config = OUTPUT_TYPE_CONFIG.get(output_type, OUTPUT_TYPE_CONFIG["meal_plan"])
        strategy = override_strategy or self.strategy

        # ── 1. Build retrieval query ──────────────────────────────────────────
        retrieval_query = self._build_retrieval_query(
            health_profile, goal, config["query_template"]
        )

        # ── 2. Retrieve context from ChromaDB ─────────────────────────────────
        chunks: list[RetrievedChunk] = []
        if rag_enabled:
            chunks = self.retriever.retrieve(
                query_text=retrieval_query,
                query_type=config["query_type"],
                n_results_per_collection=config["n_results_per_collection"],
                max_distance=0.85,
            )

        # ── 3. Build the user request string ─────────────────────────────────
        user_request = self._build_user_request(health_profile, goal, output_type)

        # ── 4. Assemble augmented prompt ──────────────────────────────────────
        if rag_enabled and chunks:
            built = self.prompt_builder.build(
                query=user_request,
                chunks=chunks,
                strategy=strategy,
                extra_system_note=extra_context,
            )
        else:
            built = self.prompt_builder.build_no_rag(query=user_request)

        # ── 5. Call vLLM ──────────────────────────────────────────────────────
        response_raw = self._call_vllm(built.system, built.user)

        # ── 6. Parse JSON response ────────────────────────────────────────────
        response_json, parse_error = self._parse_json(response_raw)

        latency = time.time() - t_start

        result = RAGResult(
            output_type=output_type,
            query_type_used=config["query_type"],
            strategy_used=built.strategy,
            n_chunks_retrieved=len(chunks),
            chunks=chunks,
            prompt_token_est=built.token_estimate,
            response_raw=response_raw,
            response_json=response_json,
            parse_error=parse_error,
            latency_seconds=round(latency, 2),
            rag_enabled=rag_enabled,
        )

        logger.info(
            f"Query complete — output_type={output_type}, "
            f"rag={rag_enabled}, chunks={len(chunks)}, "
            f"latency={latency:.1f}s, "
            f"json_ok={parse_error is None}"
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_retrieval_query(
        self, profile: dict, goal: str, template: str
    ) -> str:
        """
        Format the retrieval query template with values from the health profile.
        Falls back to empty string for missing template variables.
        """
        markers = ""
        if "blood_tests" in profile:
            markers = " ".join(
                f"{k} {v}" for k, v in profile.get("blood_tests", {}).items()
            )

        dietary_restrictions = " ".join(
            profile.get("dietary_restrictions", [])
        )

        try:
            query = template.format(
                goal=goal,
                dietary_restrictions=dietary_restrictions,
                markers=markers,
            )
        except KeyError:
            query = f"{goal} health nutrition exercise"

        return query

    def _build_user_request(
        self, profile: dict, goal: str, output_type: str
    ) -> str:
        """
        Assemble the user-facing request string from the health profile,
        appending the expected JSON schema so the model knows the output format.
        """
        schema = OUTPUT_SCHEMAS.get(output_type, "")

        profile_str = json.dumps(profile, indent=2)
        goal_line   = f"Goal: {goal}"

        output_instruction = {
            "meal_plan":    "Generate a full 7-day weekly meal plan.",
            "gym_program":  "Generate a 4-week gym training program.",
            "grocery_list": "Generate a weekly grocery list for the meal plan above.",
            "lab_analysis": "Analyse the provided blood test results.",
            "full_plan":    "Generate a complete health plan: meal plan, gym program, and grocery list.",
        }.get(output_type, "Generate a structured health plan.")

        return (
            f"Health Profile:\n{profile_str}\n\n"
            f"{goal_line}\n\n"
            f"Task: {output_instruction}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n{schema}"
        )

    def _call_vllm(self, system_prompt: str, user_message: str) -> str:
        """Send the augmented prompt to the local vLLM instance."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error("vLLM request timed out after 120s")
            raise
        except Exception as e:
            logger.error(f"vLLM call failed: {e}")
            raise

    def _parse_json(self, raw: str) -> tuple[Optional[dict], Optional[str]]:
        """
        Attempt to parse the model's response as JSON.
        Handles markdown code fences that models sometimes wrap JSON in.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            text = text.strip()

        try:
            return json.loads(text), None
        except json.JSONDecodeError as e:
            return None, str(e)