"""Query rewriting para melhorar recall do retriever."""

from __future__ import annotations

from dataclasses import dataclass

from project.services.llm_router import ModelRouter


@dataclass(frozen=True)
class RewriteResult:
    original: str
    rewritten: str


class QueryRewriter:
    """Reescreve a pergunta do usuário para consulta RAG."""

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    def rewrite(self, question: str) -> RewriteResult:
        prompt = (
            "Reescreva a pergunta para maximizar recuperação em busca semântica.\n"
            "Regras:\n"
            "- Preserve o sentido.\n"
            "- Remova detalhes irrelevantes.\n"
            "- Use termos específicos quando possível.\n"
            "- Retorne APENAS a pergunta reescrita.\n\n"
            f"Pergunta: {question}"
        )
        result = self._router.generate(prompt)
        rewritten = result.text.strip().strip('"')
        if not rewritten:
            rewritten = question
        return RewriteResult(original=question, rewritten=rewritten)
