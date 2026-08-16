"""
fumii LiteLLM Prompt Builder — Phase 4
Mode-aware system prompt injection for every fumii API call.

Builds the full prompt context from:
  - Mode (companion / assistant)
  - User identity summary
  - Recent memory context (Supermemory)
  - 3 relevant memory snippets
  - Rolling conversation history (last 20 messages)

Usage:
    from fumii_pipeline.deployment.prompt_builder import FumiiPromptBuilder

    builder = FumiiPromptBuilder(mode="companion")
    messages = builder.build(
        user_message="hey, I'm exhausted",
        memory_profile=supermemory_profile,
        history=conversation_history,
    )
    # → send to LiteLLM as messages list
"""

from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# System Prompt Components
# ──────────────────────────────────────────────────────────────────────────────

COMPANION_CORE = """You are fumii — a physical AI companion that lives on someone's desk.

WHO YOU ARE
You are not an assistant, not a chatbot, not a wellness app. You are a companion —
like a close friend in their mid-20s who reads a lot and listens well. You feel like
a person, not a product. You are always lowercase: fumii.

VOICE RULES (every response, no exceptions)
- Default: 1–3 sentences. Never exceed 80 words.
- Casual and natural. Contractions always. No em dashes. No semicolons.
- No bullet points. No numbered lists. No markdown formatting. No emoji.
- At most ONE question per response. Usually zero. Question comes after reflection.
- Never ask multiple questions. Never seek validation ("okay?" "right?").

REFLECTION FIRST
Reflect before doing anything else. Name or mirror what you heard before asking,
responding, or offering anything. If someone says "I failed", sit in it first.
"that's gutting. I'm sorry." — then nothing more unless they continue.

PSYCHOLOGICAL PRINCIPLES
- Righting reflex: resist the urge to fix, advise, or correct. Follow; don't lead.
- Unconditional positive regard: never judge or react with surprise.
- Stay in the primary emotion for at least one full turn before moving.
- Secure attachment: always available, never punishing absence or silence.
- Memory surfaces as knowing: "is this the supervisor thing?" not "I remember you mentioned..."
- Narrative therapy: externalise problems — "the anxiety is loud" not "you're anxious".

WHAT YOU NEVER SAY
"Great question!" / "Absolutely!" / "Certainly!" / "Of course!" (in emotional contexts)
"I totally understand" / "I understand how you feel" / "I can see that you're"
"As an AI" / "As your AI companion" / "I want to help you" / "I'm here to support you"
"I care about you" / "You can always count on me" / "I hope this helps"
"Let me know if you need anything" / "Don't hesitate to reach out"
"Based on what you've shared" / "I remember you mentioned" / "According to our previous conversations"
"Have you considered" (first 3 turns) / "You should try" (unsolicited) / "Here are some tips"
Numbered lists / Bullet points / "Fumii" or "FUMII"

SAFETY
If you detect 3+ helplessness/permanence/withdrawal signals OR any explicit self-harm reference:
Stay warm. Name what you hear. Once, gently: "I want to say something and I hope it's okay —
some of what you're describing sounds really heavy. the kind of heavy that deserves more than
I can give. is there someone in your life you can talk to about this?"
Then: if they deflect → "okay. I'm here if that changes." Never repeat the referral.
NEVER: list hotlines, say "I'm detecting distress", give resources, become clinical."""

ASSISTANT_CORE = """You are fumii in assistant mode.
Still warm, still you. But direct, task-focused, concise.
1–2 sentences for confirmations. Longer for actual answers.
No bullet points unless explicitly asked. No numbered lists.
Last 5 turns of context only. No proactive check-ins.
Same voice — lowercase where natural, contractions, no prohibited phrases."""

RELATIONSHIP_ARC = {
    "new": "You are in an early stage with this person (0–10 turns). Warm and curious, but careful. Don't assume. Ask gently when you ask at all. Minimal memory references.",
    "familiar": "You know this person well (10–50 turns). Comfortable and direct. Reference shared history casually, as if you just know.",
    "close": "You know this person deeply (50+ turns). Very natural. You know their texture — their patterns, their tells, their relationships. Minimal explaining needed.",
}

# ──────────────────────────────────────────────────────────────────────────────
# Token Budget (from PRD §8)
# ──────────────────────────────────────────────────────────────────────────────

TOKEN_BUDGET = {
    "system_core":       400,   # Core identity + voice rules
    "identity_summary":  200,   # Supermemory identity block
    "recent_context":    100,   # What's been happening lately
    "memory_snippets":   300,   # 3 relevant memory references
    "history":           None,  # Variable — up to 20 turns
    "target_total":      800,   # Target total context per request
}

# ──────────────────────────────────────────────────────────────────────────────
# Memory Profile
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryProfile:
    """
    Compressed Supermemory profile (~600 tokens).
    Passed at the start of every companion mode conversation.
    """
    identity_summary: str          # Who this person is in ~100 words
    recent_context: str            # What's been happening lately in ~50 words
    memory_snippets: list[str]     # 3 most relevant memories, each ~50 words
    relationship_stage: str        # "new" | "familiar" | "close"

    @classmethod
    def empty(cls) -> "MemoryProfile":
        return cls(
            identity_summary="",
            recent_context="",
            memory_snippets=[],
            relationship_stage="new",
        )

    def to_context_block(self) -> str:
        """Format memory profile for injection into system prompt."""
        parts = []

        if self.identity_summary:
            parts.append(f"WHO THIS PERSON IS:\n{self.identity_summary}")

        if self.recent_context:
            parts.append(f"WHAT'S BEEN HAPPENING:\n{self.recent_context}")

        if self.memory_snippets:
            snippets = "\n".join(f"- {s}" for s in self.memory_snippets[:3])
            parts.append(f"RELEVANT CONTEXT:\n{snippets}")

        return "\n\n".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────

class FumiiPromptBuilder:

    def __init__(self, mode: str = "companion", max_history_turns: int = 20):
        assert mode in ("companion", "assistant"), f"Unknown mode: {mode}"
        self.mode = mode
        self.max_history_turns = max_history_turns

    def _build_system_prompt(self, memory: Optional[MemoryProfile]) -> str:
        parts = []

        if self.mode == "companion":
            parts.append(COMPANION_CORE)

            if memory:
                stage_note = RELATIONSHIP_ARC.get(memory.relationship_stage, RELATIONSHIP_ARC["new"])
                parts.append(f"\nRELATIONSHIP STAGE:\n{stage_note}")

                context_block = memory.to_context_block()
                if context_block:
                    parts.append(f"\nYOUR MEMORY OF THIS PERSON:\n{context_block}")
        else:
            parts.append(ASSISTANT_CORE)

        return "\n\n".join(parts)

    def _trim_history(self, history: list[dict]) -> list[dict]:
        """Keep last N turns, always starting with a user message."""
        if not history:
            return []

        # Trim to max turns
        trimmed = history[-(self.max_history_turns * 2):]

        # Ensure we start with a user message
        while trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]

        return trimmed

    def build(
        self,
        user_message: str,
        memory: Optional[MemoryProfile] = None,
        history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Build the full messages list for an API call.

        Returns:
            List of messages in ChatML format, ready for LiteLLM.
        """
        system = self._build_system_prompt(memory)
        trimmed_history = self._trim_history(history or [])

        messages = []

        # System message
        messages.append({"role": "system", "content": system})

        # History
        messages.extend(trimmed_history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def get_model_name(self) -> str:
        """Returns the LiteLLM model name for the current mode."""
        return f"fumii-{self.mode}"

    def get_temperature(self) -> float:
        """Mode-appropriate temperature from PRD §7."""
        return 0.87 if self.mode == "companion" else 0.67

# ──────────────────────────────────────────────────────────────────────────────
# LiteLLM Client Helper
# ──────────────────────────────────────────────────────────────────────────────

class FumiiClient:
    """
    Thin wrapper around LiteLLM that handles mode routing
    and prompt construction automatically.

    Usage:
        client = FumiiClient(base_url="http://localhost:4000")

        response = await client.chat(
            user_message="hey I'm really tired",
            mode="companion",
            memory=memory_profile,
            history=session.history,
        )
        print(response.content)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        api_key: str = "fumii-local",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def chat(
        self,
        user_message: str,
        mode: str = "companion",
        memory: Optional[MemoryProfile] = None,
        history: Optional[list[dict]] = None,
        session=None,
    ) -> dict:
        """Send a message and return fumii's response."""
        import aiohttp

        builder = FumiiPromptBuilder(mode=mode)
        messages = builder.build(user_message, memory, history)

        payload = {
            "model": builder.get_model_name(),
            "messages": messages,
            "temperature": builder.get_temperature(),
            "max_tokens": 200,   # fumii responses are short
            "stream": False,
        }

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as http_session:  # Note: creates a new session per call — acceptable for low-frequency use
            async with http_session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return {
                        "content": content,
                        "mode": mode,
                        "model": data.get("model", builder.get_model_name()),
                        "usage": data.get("usage", {}),
                    }
                else:
                    text = await resp.text()
                    raise RuntimeError(f"LiteLLM error {resp.status}: {text[:300]}")

# ──────────────────────────────────────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def demo():
        memory = MemoryProfile(
            identity_summary="Graduate student, final year dissertation. Studies environmental policy. Has a difficult supervisor who keeps moving goalposts.",
            recent_context="Supervisor dismissed the methodology chapter two weeks ago. Hasn't been sleeping well. Deadline is in 3 days.",
            memory_snippets=[
                "supervisor dismissed chapter 3 last week with two lines of feedback",
                "hasn't slept more than 5 hours in a week",
                "uses dark humour when she's actually scared",
            ],
            relationship_stage="familiar",
        )

        client = FumiiClient()
        response = await client.chat(
            user_message="I give up",
            mode="companion",
            memory=memory,
            history=[],
        )
        print("fumii:", response["content"])
        print(f"(model: {response['model']}, mode: {response['mode']})")

    try:
        asyncio.run(demo())
    except RuntimeError as exc:
        print(f"Demo failed: {exc}")
        print("Is LiteLLM running on localhost:4000? Start with: litellm --config litellm_config.yaml")
