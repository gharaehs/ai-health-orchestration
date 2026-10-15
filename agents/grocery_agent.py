"""
agents/grocery_agent.py

Agent 4: Grocery

Aggregates all ingredients from the 7-day meal plan into a consolidated
weekly shopping list, grouped by category.

Strategy: one small LLM call per grocery category.
  - Collect all raw ingredient strings from the 7-day meal plan
  - Pre-bucket them by keyword into 8 categories (fast, no LLM)
  - For each non-empty category, ask the LLM to consolidate and sum quantities
  - Each call is ~20-40 items and returns a schema-constrained ConsolidatedItems
  - LLM handles fuzzy deduplication: 'chicken breast' + 'chicken, diced' → one item

A failed or truncated category call falls back to the raw items for that
category; a failed shopping-tips call falls back to default tips.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import BaseAgent
from agents.llm_client import LLMConfigError
from agents.schemas import (
    HealthProfile,
    UserGoals,
    NutritionOutput,
    GroceryOutput,
    GroceryItem,
    ConsolidatedItems,
    ShoppingTips,
)

logger = logging.getLogger(__name__)

CATEGORIES = [
    "Proteins",
    "Grains & Carbohydrates",
    "Dairy & Eggs",
    "Vegetables",
    "Fruits",
    "Fats & Oils",
    "Condiments & Seasonings",
    "Other",
]

# Keyword bucketing — fast pre-sort before LLM consolidation
CATEGORY_KEYWORDS = {
    "Proteins": [
        "chicken", "beef", "turkey", "salmon", "tuna", "cod", "egg", "tofu",
        "lentil", "chickpea", "bean", "tempeh", "shrimp", "prawn", "pork",
        "lamb", "mince", "fillet", "breast", "thigh", "steak", "fish",
    ],
    "Grains & Carbohydrates": [
        "oat", "rice", "pasta", "bread", "quinoa", "couscous", "barley",
        "flour", "wrap", "tortilla", "cracker", "cereal", "noodle", "pita",
    ],
    "Dairy & Eggs": [
        "milk", "cheese", "yogurt", "yoghurt", "cream", "butter", "egg",
        "feta", "mozzarella", "parmesan", "skyr", "kefir", "dairy",
    ],
    "Vegetables": [
        "broccoli", "spinach", "carrot", "pepper", "tomato", "onion",
        "garlic", "courgette", "zucchini", "cucumber", "lettuce", "kale",
        "cabbage", "celery", "leek", "mushroom", "aubergine", "eggplant",
        "asparagus", "pea", "green bean", "sweet potato", "potato", "vegetable",
    ],
    "Fruits": [
        "berry", "berries", "banana", "apple", "orange", "lemon", "lime",
        "mango", "grape", "avocado", "pear", "peach", "plum", "cherry",
        "strawberry", "blueberry", "raspberry", "fruit",
    ],
    "Fats & Oils": [
        "olive oil", "oil", "nut", "almond", "walnut", "cashew",
        "seed", "tahini", "peanut butter", "coconut", "flaxseed",
    ],
    "Condiments & Seasonings": [
        "salt", "pepper", "spice", "herb", "sauce", "vinegar", "mustard",
        "honey", "soy", "tamari", "cumin", "paprika", "turmeric", "oregano",
        "basil", "thyme", "cinnamon", "stock", "broth", "seasoning",
    ],
}


def _pre_bucket(ingredients: list[str]) -> dict[str, list[str]]:
    """
    Fast keyword-based pre-sort into categories.
    Each ingredient string goes into exactly one bucket.
    Unmatched items go to 'Other'.
    """
    buckets: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    for ing in ingredients:
        ing_lower = ing.lower()
        matched = False
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in ing_lower for kw in keywords):
                buckets[category].append(ing)
                matched = True
                break
        if not matched:
            buckets["Other"].append(ing)
    return buckets


DEFAULT_SHOPPING_TIPS = (
    "Buy proteins in bulk and portion into meal-sized servings before freezing.",
    "Prep grains in large batches at the start of the week.",
    "Store leafy greens with a paper towel to extend freshness.",
    "Check pantry for condiments and seasonings before shopping.",
)


class GroceryAgent(BaseAgent):

    AGENT_NAME     = "GroceryAgent"
    COLLECTION_KEY = "grocery"
    MAX_TOKENS     = 2048          # per category / shopping-tips call

    def _build_query(self, profile, goals, context):
        prefs = ", ".join(goals.dietary_preferences) if goals.dietary_preferences else "balanced"
        return f"grocery shopping food storage tips {prefs} meal prep weekly planning"

    def _build_prompt(self, profile, goals, context, rag_context):
        pass  # Not used — we override run()

    # ── Override run() ────────────────────────────────────────────────────────

    def run(self, profile, goals, context=None):
        context = context or {}
        logger.info(f"[{self.AGENT_NAME}] Starting (per-category consolidation mode)")

        nutrition: NutritionOutput | None = context.get("nutrition")
        if not nutrition:
            raise ValueError("GroceryAgent requires NutritionOutput in context")

        # ── Step 1: Collect all raw ingredient strings ────────────────────────
        all_ingredients: list[str] = []
        for day_plan in nutrition.weekly_plan:
            for meal in day_plan.meals:
                all_ingredients.extend(meal.ingredients)

        logger.info(f"[{self.AGENT_NAME}] Collected {len(all_ingredients)} ingredient entries from 7-day plan")

        # ── Step 2: Pre-bucket by keyword (no LLM, instant) ──────────────────
        buckets = _pre_bucket(all_ingredients)
        non_empty = {cat: items for cat, items in buckets.items() if items}
        logger.info(
            f"[{self.AGENT_NAME}] Pre-bucketed into {len(non_empty)} non-empty categories: "
            f"{list(non_empty.keys())}"
        )

        # ── Step 3: LLM consolidates each category separately ────────────────
        items_by_category: dict[str, list[GroceryItem]] = {cat: [] for cat in CATEGORIES}

        for category, raw_items in non_empty.items():
            logger.info(
                f"[{self.AGENT_NAME}] Consolidating '{category}' ({len(raw_items)} raw items)"
            )
            consolidated = self._consolidate_category(category, raw_items)
            items_by_category[category] = consolidated

        total_items = sum(len(items) for items in items_by_category.values())
        logger.info(f"[{self.AGENT_NAME}] Total consolidated items: {total_items}")

        # ── Step 4: Get shopping notes (one small LLM call) ──────────────────
        shopping_notes = self._get_shopping_notes(profile, goals, items_by_category)

        result = GroceryOutput(
            items_by_category=items_by_category,
            total_items=total_items,
            shopping_notes=shopping_notes,
            estimated_weekly_cost_eur=None,
        )

        logger.info(f"[{self.AGENT_NAME}] Completed successfully")
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _consolidate_category(self, category: str, raw_items: list[str]) -> list[GroceryItem]:
        """
        Ask the LLM to consolidate a list of raw ingredient strings for one
        category into deduplicated items with summed quantities.

        Example input:  ['150g chicken breast', '200g chicken breast, diced', '100g chicken thigh']
        Example output: [{'name': 'Chicken Breast', 'total_quantity': '350g'},
                         {'name': 'Chicken Thigh',  'total_quantity': '100g'}]
        """
        items_block = "\n".join(f"  - {item}" for item in raw_items)

        prompt = (
            f"You are a grocery list assistant. Consolidate this list of {category} ingredients "
            f"from a 7-day meal plan into a clean shopping list.\n\n"
            f"Raw ingredients ({category}):\n{items_block}\n\n"
            "Rules:\n"
            "- Merge similar items (e.g. 'chicken breast' and 'chicken breast, diced' → one entry).\n"
            "- Sum quantities where units match (e.g. 150g + 200g = 350g).\n"
            "- If units differ or quantity is unclear, list the larger/combined amount.\n"
            "- estimated_cost_eur: realistic European supermarket price for the total quantity.\n"
        )

        try:
            output = self._call_llm(prompt, ConsolidatedItems, self.MAX_TOKENS)
            return [
                GroceryItem(
                    name=item.name,
                    total_quantity=item.total_quantity,
                    category=category,
                    estimated_cost_eur=item.estimated_cost_eur,
                )
                for item in output.items
            ]

        except LLMConfigError:
            raise
        except Exception as e:
            logger.warning(
                f"[{self.AGENT_NAME}] Consolidation failed for '{category}': {e}. "
                f"Using raw items as fallback."
            )
            # Fallback: use raw items without consolidation
            return [
                GroceryItem(
                    name=item,
                    total_quantity="see meal plan",
                    category=category,
                    estimated_cost_eur=None,
                )
                for item in raw_items[:10]  # cap at 10 to avoid clutter
            ]

    def _get_shopping_notes(self, profile, goals, items_by_category) -> list[str]:
        """One small LLM call for practical shopping tips."""
        category_summary = []
        for cat, items in items_by_category.items():
            if items:
                names = ", ".join(i.name for i in items[:3])
                category_summary.append(f"{cat}: {names}")

        prefs     = ", ".join(goals.dietary_preferences) if goals.dietary_preferences else "none"
        allergies = ", ".join(profile.allergies) if profile.allergies else "none"

        prompt = (
            f"Shopping list for a {goals.primary_goal.value.replace('_', ' ')} goal.\n"
            f"Dietary preferences: {prefs}. Allergies: {allergies}.\n\n"
            f"Categories: {'; '.join(category_summary)}\n\n"
            "Give exactly 4 practical shopping tips."
        )

        try:
            output = self._call_llm(prompt, ShoppingTips, self.MAX_TOKENS)
            if output.tips:
                return output.tips
        except LLMConfigError:
            raise
        except Exception as e:
            logger.warning(f"[{self.AGENT_NAME}] Shopping notes call failed: {e}")

        return list(DEFAULT_SHOPPING_TIPS)
