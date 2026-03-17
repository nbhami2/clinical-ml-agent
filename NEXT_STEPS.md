\## Next Work Session



\### Ready to run - just activate venv and go:

1\. Activate venv: `.\\.venv\\Scripts\\activate`

2\. Run pipeline: `python -m src.app.run`

3\. Use this question when prompted:

&#x20;  "Across recent clinical ML prediction papers for sepsis early warning in

&#x20;  hospitalized adults, what models and validation strategies are most common,

&#x20;  how do AUROC/AUPRC compare, and what are the biggest recurring risks such

&#x20;  as data leakage, calibration, and external validation?"



\### What to expect on a successful run:

\- Vector store loads instantly (already built, 295 chunks)

\- 3 papers get extracted via Gemini

\- Evidence table + report generated

\- Output written to runs/<timestamp>/



\### After first successful run:

\- Review report.md and evidence\_table.csv in the runs/ folder

\- Then we move to Phase 2: Tool Registry + simulations (tools/registry.py)



\### Current model config:

\- Extraction: models/gemini-2.0-flash-lite

\- Report writing: models/gemini-2.0-flash

