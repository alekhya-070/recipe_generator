import os
import io
import json
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image

from gemini_client import GeminiClient, GeminiNotConfiguredError
from recipe_utils import (
    load_dataset,
    jaccard_similarity,
    filter_recipes,
    scale_ingredients_for_servings,
    compute_total_time,
    personalize_recommendations,
    normalize_ingredient_string,
)

# App configuration
st.set_page_config(page_title="Smart Recipe Generator", page_icon="🍳", layout="wide")

# Session state initialization
if "favorites" not in st.session_state:
    st.session_state.favorites = []  # store recipe dicts
if "ratings" not in st.session_state:
    st.session_state.ratings = {}    # {recipe_id: rating (1-5)}
if "last_generated" not in st.session_state:
    st.session_state.last_generated = []  # last generated recipes
if "detected_ingredients" not in st.session_state:
    st.session_state.detected_ingredients = []

# Constants / Config
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "recipes.json")
DEFAULT_SERVINGS = 2
MAX_RECIPES_TO_SHOW = 6

# Load dataset
@st.cache_data(show_spinner=False)
def get_dataset() -> List[Dict[str, Any]]:
    return load_dataset(DATASET_PATH)

dataset = get_dataset()

# Instantiate Gemini client (safe even if not configured; raises on use)
gemini_client = GeminiClient(
    api_key=os.environ.get("GEMINI_API_KEY"),
    text_model=os.environ.get("GEMINI_TEXT_MODEL", "gemini-1.5-flash"),
    vision_model=os.environ.get("GEMINI_VISION_MODEL", "gemini-1.5-flash"),
)

# Sidebar - Filters and Preferences
with st.sidebar:
    st.header("Filters & Preferences")

    dietary_prefs = st.multiselect(
        "Dietary preferences",
        options=["Vegetarian", "Vegan", "Gluten-free", "High-protein", "Low-carb"],
        help="Filter or guide recipe generation.",
    )

    difficulty = st.selectbox(
        "Cooking difficulty",
        options=["Any", "Easy", "Medium", "Hard"],
        index=0,
    )

    max_prep_time = st.slider(
        "Max prep time (minutes)", min_value=0, max_value=120, value=60, step=5
    )

    servings = st.slider(
        "Serving size", min_value=1, max_value=12, value=DEFAULT_SERVINGS, step=1,
        help="Adjust serving size. Ingredient quantities auto-scale."
    )

    st.markdown("---")
    st.caption("Gemini Configuration")
    st.text_input(
        "GEMINI_API_KEY (optional override)",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        key="gemini_key_input",
        help="Leave blank to use environment configuration.",
    )
    st.caption("Models (optional)")
    st.text_input(
        "Text Model",
        value=os.environ.get("GEMINI_TEXT_MODEL", "gemini-1.5-flash"),
        key="text_model_input",
    )
    st.text_input(
        "Vision Model",
        value=os.environ.get("GEMINI_VISION_MODEL", "gemini-1.5-flash"),
        key="vision_model_input",
    )

    if st.button("Apply Gemini Settings"):
        # Update gemini client with sidebar overrides
        gemini_client.update_config(
            api_key=st.session_state.gemini_key_input or os.environ.get("GEMINI_API_KEY"),
            text_model=st.session_state.text_model_input,
            vision_model=st.session_state.vision_model_input,
        )
        st.success("Gemini settings applied.")

# Main UI
st.title("Smart Recipe Generator")
st.write("Generate intelligent recipe suggestions from ingredients, with nutrition and filters.")

# Input section
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    ingredients_text = st.text_area(
        "Enter ingredients (comma-separated)",
        placeholder="e.g., chicken breast, garlic, olive oil, lemon, parsley",
        help="You can combine with detected ingredients from an image.",
    )

with col2:
    uploaded_image = st.file_uploader(
        "Upload image of ingredients",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Uses vision to detect ingredients."
    )
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Detecting ingredients from image..."):
                detected = gemini_client.extract_ingredients_from_image(uploaded_image.read(), uploaded_image.type)
            st.session_state.detected_ingredients = detected
            if detected:
                st.success(f"Detected ingredients: {', '.join(detected)}")
            else:
                st.warning("No ingredients confidently detected.")
        except GeminiNotConfiguredError:
            st.info("Gemini not configured. Skipping vision detection.")
        except Exception as e:
            st.error(f"Image processing failed: {e}")

combined_ingredients: List[str] = []
if ingredients_text.strip():
    combined_ingredients.extend(
        [normalize_ingredient_string(x) for x in ingredients_text.split(",") if x.strip()]
    )
if st.session_state.detected_ingredients:
    combined_ingredients.extend(st.session_state.detected_ingredients)

combined_ingredients = sorted(list({x for x in combined_ingredients if x}))  # unique & sorted

if combined_ingredients:
    st.markdown("##### Combined Ingredients")
    st.write(", ".join(combined_ingredients))

generate_btn = st.button("Generate Recipes", type="primary")

# Tabs
tab_generated, tab_favorites = st.tabs(["Generated Recipes", "My Favorites"])

def render_recipe_card(recipe: Dict[str, Any], key_prefix: str, adjustable_servings: int):
    # Adjust servings view
    base_servings = recipe.get("servings", DEFAULT_SERVINGS)
    scaled_ingredients = scale_ingredients_for_servings(recipe.get("ingredients", []), base_servings, adjustable_servings)

    st.subheader(recipe.get("name", "Recipe"))

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.caption(f"Cuisine: {recipe.get('cuisine', 'N/A')}")
        st.caption(f"Difficulty: {recipe.get('difficulty', 'N/A')}")
    with cols[1]:
        prep = recipe.get("prep_time", recipe.get("prepTime", 0))
        cook = recipe.get("cook_time", recipe.get("cookTime", 0))
        total = compute_total_time(recipe)
        st.caption(f"Prep: {prep} min | Cook: {cook} min | Total: {total} min")
    with cols[2]:
        tags = recipe.get("dietary", [])
        if tags:
            st.caption("Dietary: " + ", ".join(tags))

    st.markdown("**Ingredients**")
    for ing in scaled_ingredients:
        qty = ing.get("quantity")
        unit = ing.get("unit")
        name = ing.get("name")
        qty_str = f"{qty:g}" if isinstance(qty, (int, float)) else (qty or "")
        unit_str = f" {unit}" if unit else ""
        st.write(f"- {qty_str}{unit_str} {name}".strip())

    st.markdown("**Instructions**")
    steps = recipe.get("steps", [])
    if steps:
        for i, step in enumerate(steps, start=1):
            st.write(f"{i}. {step}")

    nutrition = recipe.get("nutrition", {})
    if nutrition:
        st.markdown("**Nutrition (per serving)**")
        ncols = st.columns(4)
        ncols[0].metric("Calories", f"{nutrition.get('calories', 'N/A')}")
        ncols[1].metric("Protein", f"{nutrition.get('protein', 'N/A')} g")
        ncols[2].metric("Fat", f"{nutrition.get('fat', 'N/A')} g")
        ncols[3].metric("Carbs", f"{nutrition.get('carbs', 'N/A')} g")

    subs = recipe.get("substitutions", [])
    if subs:
        with st.expander("Substitution Suggestions"):
            for s in subs:
                st.write(f"- {s}")

    # Interactions
    rid = recipe.get("id") or recipe.get("name")
    rating_key = f"{key_prefix}_rating_{rid}"
    current_rating = st.session_state.ratings.get(rid, 0)
    new_rating = st.slider("Rate this recipe", 1, 5, value=current_rating or 3, key=rating_key)
    st.session_state.ratings[rid] = new_rating

    fav_key = f"{key_prefix}_fav_{rid}"
    is_fav = any((r.get("id") or r.get("name")) == rid for r in st.session_state.favorites)
    if st.toggle("Save to Favorites", value=is_fav, key=fav_key):
        if not is_fav:
            st.session_state.favorites.append(recipe)
    else:
        if is_fav:
            st.session_state.favorites = [r for r in st.session_state.favorites if (r.get("id") or r.get("name")) != rid]

    st.divider()

def safe_generate_with_gemini(ingredients: List[str], prefs: List[str], max_prep: int, diff: str, servings: int) -> List[Dict[str, Any]]:
    try:
        return gemini_client.generate_recipes_from_ingredients(
            ingredients=ingredients,
            dietary_prefs=prefs,
            max_prep_time=max_prep,
            difficulty=None if diff == "Any" else diff,
            servings=servings,
            n_recipes=5
        )
    except GeminiNotConfiguredError:
        st.info("Gemini not configured. Falling back to local dataset recommendations.")
        return []
    except Exception as e:
        st.warning(f"Recipe generation failed: {e}")
        return []

def dataset_suggestions(ingredients: List[str], prefs: List[str], max_prep: int, diff: str) -> List[Dict[str, Any]]:
    # Rank dataset by similarity to provided ingredients + filter
    filtered = filter_recipes(dataset, dietary=prefs, difficulty=None if diff == "Any" else diff, max_prep_time=max_prep)
    if not ingredients:
        return filtered[:MAX_RECIPES_TO_SHOW]

    scored = []
    ing_set = set(normalize_ingredient_string(x) for x in ingredients)
    for r in filtered:
        recipe_ing = set(normalize_ingredient_string(i["name"]) for i in r.get("ingredients", []))
        score = jaccard_similarity(ing_set, recipe_ing)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:MAX_RECIPES_TO_SHOW]]

with tab_generated:
    if generate_btn:
        if not combined_ingredients:
            st.warning("Please provide ingredients via text or image.")
        else:
            with st.spinner("Generating recipes..."):
                ai_recipes = safe_generate_with_gemini(combined_ingredients, dietary_prefs, max_prep_time, difficulty, servings)
                ds_recipes = dataset_suggestions(combined_ingredients, dietary_prefs, max_prep_time, difficulty)

                # Merge results, prioritizing AI recipes
                merged: List[Dict[str, Any]] = []
                seen_ids = set()
                for r in ai_recipes + ds_recipes:
                    rid = r.get("id") or r.get("name")
                    if rid not in seen_ids:
                        merged.append(r)
                        seen_ids.add(rid)

                st.session_state.last_generated = merged[:MAX_RECIPES_TO_SHOW]

    if st.session_state.last_generated:
        st.markdown("### Results")
        for idx, recipe in enumerate(st.session_state.last_generated):
            render_recipe_card(recipe, key_prefix=f"gen_{idx}", adjustable_servings=servings)
    else:
        st.info("No recipes yet. Enter ingredients and click Generate.")

with tab_favorites:
    if st.session_state.favorites:
        st.markdown("### My Favorites")
        for idx, recipe in enumerate(st.session_state.favorites):
            render_recipe_card(recipe, key_prefix=f"fav_{idx}", adjustable_servings=servings)

        # Personalized suggestions
        st.markdown("### Personalized Suggestions")
        suggestions = personalize_recommendations(
            dataset=dataset,
            favorites=st.session_state.favorites,
            ratings=st.session_state.ratings,
            dietary_prefs=dietary_prefs,
            top_k=3
        )
        if suggestions:
            for idx, recipe in enumerate(suggestions):
                render_recipe_card(recipe, key_prefix=f"pers_{idx}", adjustable_servings=servings)
        else:
            st.info("No personalized suggestions available yet.")
    else:
        st.info("You have no favorites yet. Save some recipes to see personalized suggestions.")
