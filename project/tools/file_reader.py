"""Tool para leitura de arquivos locais com restrição de diretório."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from project.config.settings import Settings
from project.tools.registry import ToolContext, register_tool_factory


class FileReadInput(BaseModel):
    """Input schema para leitura de arquivos."""

    path: str = Field(..., description="Caminho do arquivo (relativo ou absoluto) a ser lido.")
    max_chars: int | None = Field(default=None, description="Limite de caracteres para retornar.")


class FileReadTool(BaseTool):
    """Lê arquivo de texto local respeitando um root permitido."""

    name: str = "read_file"
    description: str = "Lê conteúdo de um arquivo local (texto) sob um diretório permitido."
    args_schema: type[BaseModel] = FileReadInput

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings

    def _run(self, path: str, max_chars: int | None = None, **kwargs) -> str:
        requested = Path(path)
        resolved = (requested if requested.is_absolute() else (Path.cwd() / requested)).resolve()

        allowed_root = self._settings.allow_local_file_reads_under.resolve()
        if allowed_root not in resolved.parents and resolved != allowed_root:
            raise ValueError(f"Acesso negado. O caminho deve estar sob {allowed_root}.")

        if not resolved.exists() or not resolved.is_file():
            raise ValueError("Arquivo não encontrado.")

        limit = max_chars or self._settings.max_tool_output_chars
        content = resolved.read_text(encoding="utf-8", errors="ignore")
        if len(content) > limit:
            return content[:limit] + "\n\n[TRUNCADO]"
        return content


def _factory(ctx: ToolContext) -> FileReadTool:
    return FileReadTool(ctx.settings)


register_tool_factory("read_file", _factory)
