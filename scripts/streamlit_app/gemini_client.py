import base64
import json
from typing import Any, Dict, List, Optional

import google.generativeai as genai

class GeminiNotConfiguredError(Exception):
    pass

class GeminiClient:
    def __init__(self, api_key: Optional[str], text_model: str = "gemini-1.5-flash", vision_model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.text_model = text_model
        self.vision_model = vision_model
        self._configured = False
        self._configure()

    def _configure(self):
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self._configured = True
        else:
            self._configured = False

    def update_config(self, api_key: Optional[str], text_model: Optional[str] = None, vision_model: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key
        if text_model:
            self.text_model = text_model
        if vision_model:
            self.vision_model = vision_model
        self._configure()

    def _ensure_configured(self):
        if not self._configured:
            raise GeminiNotConfiguredError("GEMINI_API_KEY is not configured.")

    def extract_ingredients_from_image(self, image_bytes: bytes, media_type: str = "image/jpeg") -> List[str]:
        """
        Use Gemini Vision to detect ingredients in an image.
        Returns a list of ingredient names.
        """
        self._ensure_configured()

        model = genai.GenerativeModel(self.vision_model)

        # Gemini expects "file" parts; the SDK supports raw bytes when using dict with data and mime type.
        prompt = (
            "You are an expert ingredient detector. Identify visible food ingredients in the photo. "
            "Return JSON only in this exact shape:\n"
            '{ "ingredients": ["ingredient1", "ingredient2", ...] }.\n'
            "Use common grocery names (singular), lowercase. Exclude utensils, packaging, or background items."
        )
        response = model.generate_content(
            [
                {"mime_type": media_type, "data": image_bytes},
                {"text": prompt},
            ]
        )

        text = response.text or ""
        # Attempt to parse out JSON
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(text[start:end+1])
                ingredients = obj.get("ingredients", [])
                return [str(x).strip().lower() for x in ingredients if str(x).strip()]
        except Exception:
            pass

        # Fallback: naive parsing by lines/commas
        candidates: List[str] = []
        for token in text.replace("\n", ",").split(","):
            token = token.strip().lower()
            if token and len(token.split()) <= 3 and token.isascii():
                candidates.append(token)
        # Deduplicate and keep reasonable list
        uniq = sorted(list({c for c in candidates if c and c.isascii()}))
        return uniq[:20]

    def generate_recipes_from_ingredients(
        self,
        ingredients: List[str],
        dietary_prefs: List[str],
        max_prep_time: int,
        difficulty: Optional[str],
        servings: int,
        n_recipes: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Use Gemini Text to generate multiple recipes with structure.
        Returns a list of recipe dicts.
        """
        self._ensure_configured()

        model = genai.GenerativeModel(self.text_model)

        dietary_str = ", ".join(dietary_prefs) if dietary_prefs else "none"
        difficulty_str = difficulty or "Any"

        system = (
            "You generate structured recipes. Always return pure JSON with a top-level 'recipes' array. "
            "Each recipe must include: id (slug), name, cuisine, difficulty, servings, prep_time, cook_time, "
            "dietary (array), ingredients (array of {name, quantity, unit}), steps (array of strings), "
            "nutrition ({calories, protein, fat, carbs}), substitutions (array of strings). "
            "Quantities must be numeric where possible. Times in minutes."
        )

        user = f"""
Ingredients available: {ingredients}
Dietary preferences: {dietary_str}
Target difficulty: {difficulty_str}
Max prep time: {max_prep_time} minutes
Target servings: {servings}
Please generate {n_recipes} diverse, realistic recipes that use the ingredients where possible. 
Offer reasonable substitutions for missing items. Use clear, concise steps.
Return JSON only.
"""

        response = model.generate_content([{"text": system}, {"text": user}])
        text = response.text or ""

        # Extract JSON
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                obj = json.loads(text[start:end+1])
                recipes = obj.get("recipes", [])
                # Basic normalization
                norm: List[Dict[str, Any]] = []
                for r in recipes:
                    # Ensure required fields exist
                    r.setdefault("id", r.get("name", "").lower().replace(" ", "-")[:60])
                    r.setdefault("servings", servings)
                    r.setdefault("dietary", [])
                    r.setdefault("ingredients", [])
                    r.setdefault("steps", [])
                    r.setdefault("nutrition", {})
                    r.setdefault("prep_time", r.get("prepTime", 0))
                    r.setdefault("cook_time", r.get("cookTime", 0))
                    # Coerce ingredient structure
                    new_ings = []
                    for ing in r.get("ingredients", []):
                        name = (ing.get("name") if isinstance(ing, dict) else str(ing)).strip().lower()
                        qty = ing.get("quantity") if isinstance(ing, dict) else None
                        unit = ing.get("unit") if isinstance(ing, dict) else None
                        new_ings.append({"name": name, "quantity": qty, "unit": unit})
                    r["ingredients"] = new_ings
                    norm.append(r)
                return norm
        except Exception:
            pass

        # Fallback: return empty (caller will merge with dataset)
        return []
