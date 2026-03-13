"""Avaliação simples do agente usando um dataset JSON."""

from __future__ import annotations

import json
from pathlib import Path

from project.app import AssistantManager
from project.config.settings import load_settings
from project.security.prompt_guard import PromptGuard
from project.services.logging_service import setup_logging


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_settings(project_root=root)
    logger = setup_logging(settings.logs_dir)
    manager = AssistantManager(settings=settings, logger=logger)
    assistant = manager.get(session_id="eval", streaming=False)
    guard = PromptGuard(settings)

    dataset_path = Path(__file__).with_name("dataset.json")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    results: list[dict] = []
    for row in data:
        q = str(row["question"])
        expected = [str(x) for x in row.get("expected_answer_contains", [])]
        out = assistant.invoke(guard.enforce(q))
        answer = str(out.get("output", ""))
        ok = all(e in answer for e in expected) if expected else True
        results.append({"question": q, "ok": ok, "output": answer[:5000]})

    report = Path(settings.logs_dir) / "agent_eval_report.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for r in results if r["ok"])
    print(f"agent_eval: {passed}/{len(results)} passed. Report: {report}")


if __name__ == "__main__":
    main()
