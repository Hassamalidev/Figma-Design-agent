"""Turns the instruction into an ordered plan: tokens, then components, then composition."""
from __future__ import annotations

import json
import logging
import re

from agent.llm import ModelClient
from agent.prompts import (
    ENHANCE_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    enhance_user_message,
    planning_user_message,
)
from agent.state import RunState

logger = logging.getLogger(__name__)

# Used only if the model's response can't be parsed as JSON -- keeps the loop
# moving instead of crashing the whole run on a formatting slip.
_FALLBACK_STEP = "Build the requested design as a single composition step."


def enhance_instruction(instruction: str, llm: ModelClient) -> str:
    """Expand a short instruction into a detailed creative brief before planning.

    Keeps the user's explicit requirements intact and fills in everything
    they left unspecified with concrete design judgment (palette, layout,
    type) -- the goal is a plan built from a real design brief, not a raw
    one-liner. Falls back to the raw instruction if the model returns nothing
    usable, so a bad enhancement never blocks the run.
    """
    logger.info("Writing a design brief for: %s", instruction)
    messages = [
        {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
        {"role": "user", "content": enhance_user_message(instruction)},
    ]
    message = llm.complete(messages, tools=None)
    brief = (message.content or "").strip()
    if not brief:
        logger.info("Brief came back empty -- building from the raw instruction instead.")
        return instruction
    logger.info("Design brief:\n%s", brief)
    return brief


def make_plan(instruction: str, state: RunState, llm: ModelClient) -> list[str]:
    """Ask the model for an ordered list of small, atomic build steps."""
    messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": planning_user_message(
                instruction, state.inspection_summary, state.existing_sections
            ),
        },
    ]
    message = llm.complete(messages, tools=None)
    steps = _parse_steps(message.content or "") or [_FALLBACK_STEP]
    logger.info("Plan (%d step%s):\n%s", len(steps), "" if len(steps) == 1 else "s", _format_plan(steps))
    return steps


def _format_plan(steps: list[str]) -> str:
    return "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, start=1))


def _parse_steps(content: str) -> list[str]:
    """Parse a JSON array of strings out of the model's reply, tolerating stray fences/prose."""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(step).strip() for step in parsed if str(step).strip()]
