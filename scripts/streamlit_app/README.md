# Smart Recipe Generator (Streamlit + Gemini)

An intelligent recipe recommendation app that:
- Accepts text or image ingredients
- Uses Gemini Vision to detect ingredients in images
- Uses Gemini Text to generate multiple recipe suggestions
- Includes nutrition, substitutions, filters (dietary, difficulty, prep time)
- Supports serving-size scaling, favorites, ratings, and simple personalization
- Falls back to a local dataset if Gemini is not configured

## Project Structure
- app.py — Streamlit UI and orchestration
- gemini_client.py — Gemini vision/text wrappers
- recipe_utils.py — Filtering, scaling, similarity, personalization
- data/recipes.json — 20+ recipe dataset
- requirements.txt — Dependencies

## Environment Variables
- GEMINI_API_KEY — required for Gemini features
- GEMINI_TEXT_MODEL — optional (default: gemini-1.5-flash)
- GEMINI_VISION_MODEL — optional (default: gemini-1.5-flash)

In Streamlit Cloud, set Secrets: GEMINI_API_KEY. Locally, you can `export GEMINI_API_KEY=...`.

## Run Locally
1. Python 3.10+
2. Install deps: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Open the provided local URL.

## Deployment

### Streamlit Cloud (recommended, free)
- Push this folder to a GitHub repo
- Create a new app on Streamlit Cloud pointing to `scripts/streamlit_app/app.py`
- Set Secrets: `GEMINI_API_KEY`

### Heroku
- Add the app with Python buildpack
- Set config vars: `GEMINI_API_KEY`
- Use `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0` in Procfile

### Vercel / Netlify
These platforms are optimized for Node/SSR. For Streamlit (Python), use:
- A containerized deployment, or
- Prefer Streamlit Cloud/HF Spaces for simplest Python hosting.

## Notes
- If GEMINI_API_KEY is missing, the app gracefully falls back to the local dataset for suggestions.
- Ingredient recognition and generation quality improve with high-quality images and clear inputs.
