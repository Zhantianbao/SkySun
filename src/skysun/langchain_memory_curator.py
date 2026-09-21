"""LangChain/DeepSeek adapter for SkySun's structured Memory Curator."""

import json

from langchain_core.language_models.chat_models import BaseChatModel

from skysun.memory_curator import CurationContext, MemoryPlan

_CURATOR_PROMPT = (
    "You are SkySun's private memory curator. Return a JSON object with a decisions array matching the MemoryPlan schema; "
    "return no prose and never answer the user. Consider durable, future-useful facts stated by the USER in "
    "the CURRENT message: stable preferences, long-term goals, recurring constraints, "
    "project facts, explicit remember requests, clear replacements, and explicit forget "
    "requests. Ignore transient states, passing activities, one-off tasks, speculation, "
    "hypotheticals, and assistant-generated assumptions. Be conservative: an empty "
    "decisions list means ignore. Never store passwords, API keys, access tokens, "
    "private keys, authentication credentials, or database passwords. Use existing "
    "semantic keys for updates and avoid duplicate facts. For a clear replacement, "
    "upsert the existing key with the new fact. Delete only on an explicit forget "
    "request. Return no more than three decisions. Each upsert needs a stable "
    "snake_case key and concise content. The supplied history is context only; do "
    "not independently mine old turns for new facts."
)


class LangChainMemoryCurator:
    def __init__(self, model: BaseChatModel) -> None:
        self._structured_model = model.with_structured_output(
            MemoryPlan, method="json_mode"
        )

    def decide(self, context: CurationContext) -> MemoryPlan:
        memories = [
            {"key": memory.key, "content": memory.content, "category": memory.category}
            for memory in context.current_memories
        ]
        history = [
            {"role": turn.role, "content": turn.content}
            for turn in context.recent_history
        ]
        payload = {
            "conversation_id": str(context.conversation_id),
            "current_user_message": context.user_message,
            "recent_history": history,
            "current_personal_memories": memories,
        }
        # The provider's structured-output adapter validates the Pydantic schema.
        result = self._structured_model.invoke(
            [
                ("system", _CURATOR_PROMPT),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        return MemoryPlan.model_validate(result)
