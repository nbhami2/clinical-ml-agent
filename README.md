# clinical-ml-agent
An autonomous research assistant for analyzing clinical machine learning prediction papers.

Given a research question, the system:
 - Retrieves relevant papers (RAG)
 - Extracts methodological details
 - Synthesizes cross-paper comparisons
 - Proposes experiment designs
 - Runs statistical simulations 
 - Generates a citation-backed report
 - Uses a planner and reflection loop for self-correction


Motivation:
Clinical ML literature generally suffers from:
 - Inconsitent validation strategies
 - Data leakage risks
 - Missing calibration reporting
 - Lack of external validation
 - Poor reproducibility

This project builds an agentic system that: 
- Standardizes extraction across studies
- Detects methodological weaknesses
- Proposes stronger evaluation designs
- Maintains full traceability of reasoning and tool calls.


Architecture:
Research Question -> Planner -> Tool registry (literature retrieval, paper extraction, statistical simulation tools, report writer) -> Memory store (Vector DB + Artifacts) -> Refelction Loop (quality + citation checks) -> Evaluation Harness


Getting Started (Windows):
1. Clone Repo

```git clone https://github.com/<your-username>/clinical-ml-agent.git \n```
```cd clinical-ml-agent```

2. Create Virtual Environment
uv venv
.\.venv\Scripts\activate
uv sync --dev
3. Run the App
python -m src.app.run

You will be prompted for a research question.

Example:
```Across recent sepsis early warning ML papers, what validation strategies are most common?```

The system will generate:

```runs/<timestamp>/
├── report.md
├── trace.json
├── retrieved.json
└── extractions.json

Disclaimer:
This system is a research analysis tool. It does *not* provide clinical recommendations or medical advice.