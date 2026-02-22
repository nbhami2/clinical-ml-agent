from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    question = input("Research question: ").strip()
    if not question:
        raise SystemExit("No question provided.")

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "run_id": run_id,
        "question": question,
        "steps": [],
        "artifacts": [],
    }

    # Placeholder: later this will call planner/executor/tools
    report_md = f"# Clinical ML Research Assistant Report\n\n## Question\n{question}\n\n## Notes\nMVP scaffold.\n"

    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    print(f"\nWrote:\n- {out_dir / 'report.md'}\n- {out_dir / 'trace.json'}")


if __name__ == "__main__":
    main()