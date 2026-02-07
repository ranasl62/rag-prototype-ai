from __future__ import annotations

from typing import List

from rag_app.llm import chat
from rag_app.models import AgentResponse, AgentStep, SourceChunk
from rag_app.rag import rag_answer


class AgenticAggregator:
    def __init__(self) -> None:
        self.system_prompt = (
            "You are a thoughtful research assistant. You plan, retrieve, draft, "
            "critique, and finalize answers while keeping a natural tone."
        )

    def _plan(self, question: str, llm_provider: str | None, llm_model: str | None) -> str:
        return chat(
            self.system_prompt,
            f"Create a short plan (2-3 steps) to answer: {question}",
            provider=llm_provider,
            model_name=llm_model,
        )

    def _critique(
        self, draft: str, question: str, llm_provider: str | None, llm_model: str | None
    ) -> str:
        return chat(
            self.system_prompt,
            "Critique the draft for missing info or unsupported claims. "
            f"Question: {question}\nDraft:\n{draft}",
            provider=llm_provider,
            model_name=llm_model,
        )

    def _revise(
        self, draft: str, critique: str, llm_provider: str | None, llm_model: str | None
    ) -> str:
        return chat(
            self.system_prompt,
            "Revise the draft using the critique. Keep it clear and human.\n"
            f"Draft:\n{draft}\n\nCritique:\n{critique}",
            provider=llm_provider,
            model_name=llm_model,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
        include_steps: bool = True,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> AgentResponse:
        steps: List[AgentStep] = []
        if include_steps:
            plan = self._plan(question, llm_provider, llm_model)
            steps.append(AgentStep(name="plan", detail=plan))

        draft, sources = rag_answer(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        if include_steps:
            steps.append(
                AgentStep(name="retrieve_and_draft", detail="Generated draft from RAG context.")
            )

        critique = (
            self._critique(draft, question, llm_provider, llm_model) if include_steps else ""
        )
        if include_steps:
            steps.append(AgentStep(name="critique", detail=critique))

        final = (
            self._revise(draft, critique, llm_provider, llm_model) if include_steps else draft
        )
        if include_steps:
            steps.append(AgentStep(name="finalize", detail="Applied critique to produce final answer."))

        source_chunks = [
            SourceChunk(source=src, title=title, doc_id=doc_id, page=page, content=content)
            for src, title, doc_id, page, content in sources
        ]
        return AgentResponse(answer=final, sources=source_chunks, steps=steps)
