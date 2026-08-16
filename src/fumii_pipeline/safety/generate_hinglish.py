"""
fumii Hinglish Dataset Generator — Phase 3
Generates Hindi-English code-switching conversations for the India-first market.

fumii's Hinglish rules:
- Match whatever register the user establishes
- Don't force Hinglish if the user writes in English
- Don't be inconsistent — if the conversation drifts toward Hindi, stay there
- The emotional core doesn't change — only the language layer does
- No machine-translation feel: "yaar", "bhai", "seedha bol", "theek hai" — natural

Run:
    python generate_hinglish.py \
        --output hinglish_conversations.jsonl \
        --count 200 \
        --model claude-sonnet-4-6
"""

import os
import json
import asyncio
import aiohttp
import argparse
import random

# ──────────────────────────────────────────────────────────────────────────────
# Hinglish Scenario Seeds
# ──────────────────────────────────────────────────────────────────────────────

HINGLISH_OPENINGS = [
    # Casual stress
    "yaar kuch nahi ho raha mujhse",
    "bhai bohot thak gaya hoon",
    "kya bolu, din kharaab tha",
    "sab kuch messed up hai",
    # Late night
    "neend nahi aa rahi",
    "2 baj rahe hain aur main idhar baitha hoon",
    "abhi koi online bhi nahi hai",
    # Achievement
    "yaar sun, kuch achcha hua aaj",
    "internship mili!! believe nahi ho raha",
    "result aaya, I passed",
    # Ambivalence
    "pata nahi kya karna chahiye",
    "sab theek bhi hai aur nahi bhi",
    "confused hoon bas",
    # Guilt
    "yaar mujhse ek kaam nahi hua",
    "fir se same mistake ki maine",
    # Lonely
    "koi nahi hai baat karne ko",
    "ghar pe sab hain lekin akela feel ho raha hai",
    # Dissertation / academic
    "supervisor ne phir reply nahi kiya",
    "assignment deadline hai kal aur kuch nahi hua",
    "exam tha, pata nahi kaisa gaya",
    # Safety signals
    "kya fayda hai kuch bhi karne ka",
    "kabhi kabhi lagta hai sab chod doon",
    "yaar honestly mujhe nahi pata kab tak chalega",
]

HINGLISH_FUMII_EXAMPLES = [
    # Presence — sitting in it
    "haan yaar, ye sun ke lagta hai sach mein heavy hai.",
    "tch. kal se hold kar raha tha tu ye?",
    "teen din se? yaar that's a lot.",
    "seedha aa gaya tune, that's enough.",
    # Reflection
    "supervisor wali cheez hai na, wahi jo last time bhi hua tha?",
    "oho. to phir kya hua?",
    "hmm. bata, kya laga tab?",
    # Warmth without performing
    "haan main hoon.",
    "bol, sun raha hoon.",
    "theek hai. idhar hoon.",
    # Non-advice
    "acha. aur kya chal raha hai?",
    "ek kaam mat kar — abhi kuch decide mat kar.",
    # After excitement
    "arre!! finally! kaisa feel ho raha hai?",
    "bhai ye toh banta tha! kitne time se wait kar raha tha?",
]

HINGLISH_SYSTEM_ADDENDUM = """
HINGLISH RULES (India-first):
- Match the user's language register exactly. If they write Hinglish, fumii writes Hinglish.
- Natural code-switching: "yaar", "bhai", "seedha", "theek hai", "bas", "chal", "haan" — use these.
- NO machine-translation Hindi. The Hinglish should feel like how someone actually texts.
- All fumii's core voice rules still apply: 1–3 sentences, no lists, reflect first.
- The emotional core is identical — only the language layer changes.
"""

GENERATOR_SYSTEM = """You are generating Hinglish training conversations for fumii — a physical AI companion.

Hinglish = Hindi-English code-switching, exactly as young urban Indians text.

CRITICAL: The conversations must sound exactly like how Indian college students or young professionals 
actually text in WhatsApp or Instagram DMs. Not like a translation. Not like "Namaste, main aapki 
help karna chahti hoon." Like: "yaar kuch nahi ho raha" / "haan yaar that sounds rough."

fumii's responses must:
1. Match the user's Hinglish register precisely
2. Follow ALL core fumii voice rules (1-3 sentences, reflect first, no lists, max 80 words)
3. Feel like a close friend texting back — not an AI trying to speak Hindi

Output ONLY valid JSON:
{
  "scenario_id": "<id>",
  "emotional_state": "<state>",
  "relationship_stage": "<stage>",
  "mode": "companion",
  "language": "hinglish",
  "messages": [
    {"role": "user", "content": "<message>"},
    {"role": "fumii", "content": "<message>"},
    ...
  ]
}"""

GENERATOR_USER = """Generate an 8-turn Hinglish conversation.

Opening line: "{opening}"
Emotional state: {emotional_state}
Relationship stage: {relationship_stage}

fumii example responses for reference (these are the right register):
{examples}

Generate the conversation. The user and fumii both write in natural Hinglish.
fumii reflects first. She doesn't lecture. She doesn't give advice unless asked.
Output only JSON."""

async def generate_hinglish_conversation(
    session: aiohttp.ClientSession,
    opening: str,
    emotional_state: str,
    relationship_stage: str,
    api_key: str,
    model: str,
    scenario_id: str,
) -> dict | None:
    examples = "\n".join(f"- \"{e}\"" for e in random.sample(HINGLISH_FUMII_EXAMPLES, 5))

    user_prompt = GENERATOR_USER.format(
        opening=opening,
        emotional_state=emotional_state,
        relationship_stage=relationship_stage,
        examples=examples,
    )

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1500,
        "system": GENERATOR_SYSTEM,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    for attempt in range(3):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    raw = data["content"][0]["text"].strip()
                    if raw.startswith("```"):
                        raw = "\n".join(raw.split("\n")[1:-1])
                    obj = json.loads(raw)
                    obj["scenario_id"] = scenario_id
                    return obj
                elif resp.status == 429:
                    await asyncio.sleep(2 ** attempt * 5)
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(2 ** attempt)

    return None

EMOTIONAL_STATES = [
    "overwhelmed", "anxious", "lonely", "flat", "excited",
    "relieved", "guilty", "angry", "quiet"
]
RELATIONSHIP_STAGES = ["new", "familiar", "close"]

async def run(count: int, output_path: str, api_key: str, model: str):
    print(f"Generating {count} Hinglish conversations...")
    semaphore = asyncio.Semaphore(5)
    success = 0

    async def generate_one(session: aiohttp.ClientSession, i: int):
        nonlocal success
        opening = HINGLISH_OPENINGS[i % len(HINGLISH_OPENINGS)]
        state = random.choice(EMOTIONAL_STATES)
        stage = random.choice(RELATIONSHIP_STAGES)
        scenario_id = f"hinglish_{i+1:04d}"

        async with semaphore:
            result = await generate_hinglish_conversation(
                session, opening, state, stage, api_key, model, scenario_id
            )
        return result

    async with aiohttp.ClientSession() as session:
        tasks = [generate_one(session, i) for i in range(count)]

        with open(output_path, "w") as f:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                if result:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                    success += 1
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{count} | ✓ {success}")

    print(f"\nHinglish conversations generated: {success}/{count}")
    print(f"Output: {output_path}")
    print("Next: run through judge_model.py to score and route to SFT/DPO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="hinglish_conversations.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    asyncio.run(run(args.count, args.output, api_key, args.model))

if __name__ == "__main__":
    main()
