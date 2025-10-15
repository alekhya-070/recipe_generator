import json
import math
from typing import Any, Dict, List, Optional, Set

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_ingredient_string(s: str) -> str:
    return " ".join(s.strip().lower().split())

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    if not union:
        return 0.0
    return len(inter) / len(union)

def filter_recipes(
    recipes: List[Dict[str, Any]],
    dietary: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    max_prep_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    results = []
    dset = set(x.lower() for x in (dietary or []))

    for r in recipes:
        # Dietary
        if dset:
            tags = set(x.lower() for x in r.get("dietary", []))
            if not dset.issubset(tags):
                continue
        # Difficulty
        if difficulty and difficulty.lower() != "any":
            if r.get("difficulty", "").lower() != difficulty.lower():
                continue
        # Prep time
        if max_prep_time is not None:
            prep = r.get("prep_time", r.get("prepTime", 0)) or 0
            if prep > max_prep_time:
                continue

        results.append(r)

    return results

def scale_ingredients_for_servings(ingredients: List[Dict[str, Any]], base_servings: int, target_servings: int) -> List[Dict[str, Any]]:
    if not base_servings or base_servings <= 0:
        base_servings = 1
    factor = target_servings / float(base_servings)
    scaled = []
    for ing in ingredients:
        qty = ing.get("quantity")
        name = ing.get("name")
        unit = ing.get("unit")
        if isinstance(qty, (int, float)):
            new_qty = round(qty * factor, 2)
        else:
            new_qty = qty  # leave non-numeric as-is
        scaled.append({"name": name, "quantity": new_qty, "unit": unit})
    return scaled

def compute_total_time(recipe: Dict[str, Any]) -> int:
    prep = recipe.get("prep_time", recipe.get("prepTime", 0)) or 0
    cook = recipe.get("cook_time", recipe.get("cookTime", 0)) or 0
    return int(prep) + int(cook)

def average_rating_for_recipe(recipe_id: str, ratings: Dict[str, int]) -> float:
    if recipe_id in ratings:
        return float(ratings[recipe_id])
    return 0.0

def personalize_recommendations(
    dataset: List[Dict[str, Any]],
    favorites: List[Dict[str, Any]],
    ratings: Dict[str, int],
    dietary_prefs: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Simple personalization:
    - Compute preferred dietary tags and cuisines from favorites with higher ratings.
    - Rank candidates from dataset accordingly.
    """
    if not favorites:
        return []

    # Include "south indian" and "healthy" (if applicable) in dietary preferences for stronger bias
    diet_pref_set = set(x.lower() for x in dietary_prefs)
    diet_pref_set.add("south indian")
    # You might want to add "healthy" if you explicitly tag recipes as such in your dataset or Gemini output.
    # For now, relying on the prompt to Gemini for "mostly healthy".

    cuisine_scores = {}
    dietary_scores = {}

    for r in favorites:
        rid = r.get("id") or r.get("name")
        score = ratings.get(rid, 3)  # default neutral
        cuisine = r.get("cuisine", "").lower()
        if cuisine:
            cuisine_scores[cuisine] = cuisine_scores.get(cuisine, 0) + score
        for tag in r.get("dietary", []):
            dietary_scores[tag.lower()] = dietary_scores.get(tag.lower(), 0) + score

    candidates = []
    seen = set((r.get("id") or r.get("name")) for r in favorites)
    for r in dataset:
        rid = r.get("id") or r.get("name")
        if rid in seen:
            continue
        # Basic dietary match
        tags = set(x.lower() for x in r.get("dietary", []))
        diet_match = len(diet_pref_set & tags) # Enhanced to consider "south indian" and other healthy tags
        # Cuisine affinity
        cscore = cuisine_scores.get(r.get("cuisine", "").lower(), 0)
        # Dietary affinity
        dscore = sum(dietary_scores.get(t.lower(), 0) for t in r.get("dietary", []))
        total = diet_match * 2 + cscore + dscore
        
        # Further boost if it's explicitly South Indian
        if "south indian" in tags:
            total += 5 # Arbitrary boost value

        candidates.append((total, r))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in candidates[:top_k]]
