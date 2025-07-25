import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")

# Shared LLM instance for o4‑mini‑high
_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=API_KEY)

def parse_request(user_input: str) -> dict:
    """Extract item_id or item_name and content_type from a free‑form user input."""
    prompt = f"""
You are a parser that extracts exactly two things from the user's request:
1) item_id (e.g. "P012") or item_name (e.g. "Red Apple")
2) content_type: one of ["Social Media", "Website Listing", "Video Ad"]

Return a JSON object with keys "item_id", "item_name", "content_type".
If a field is not present, set it to null.

Examples:
Input: "Please create a social media post for P012"
Output: {{ "item_id": "P012", "item_name": null, "content_type": "Social Media" }}

Input: "Make a video ad for Red Delicious Apple"
Output: {{ "item_id": null, "item_name": "Red Delicious Apple", "content_type": "Video Ad" }}

Now parse this:
Input: {json.dumps(user_input)}
"""
    resp = _llm([HumanMessage(content=prompt)]).content.strip()
    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        # safe fallback
        return {"item_id": None, "item_name": None, "content_type": None}

def social_media_agent(product: dict) -> str:
    prompt = f"""
Generate a catchy social media post (max 280 chars) for the product below:

• Name: {product['name']}
• Features: {product['features']}
• Price: {product['price']}
• USP: {product['usp']}

Tone: Friendly & persuasive.
"""
    return _llm([HumanMessage(content=prompt)]).content.strip()

def website_listing_agent(product: dict) -> str:
    prompt = f"""
Write an e‑commerce listing (3–4 sentences) for:

• Name: {product['name']}
• Features: {product['features']}
• Specs: {product['specs']}
• USP: {product['usp']}

Tone: Informative & trustworthy.
"""
    return _llm([HumanMessage(content=prompt)]).content.strip()

def video_ad_agent(product: dict) -> str:
    prompt = f"""
Draft a 30‑second video ad script with intro, hook, features & CTA:

• Name: {product['name']}
• Key Selling Points: {product['usp']}
• Target Audience: {product['audience']}

Keep it snappy!
"""
    return _llm([HumanMessage(content=prompt)]).content.strip()
