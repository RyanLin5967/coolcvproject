# Roboflow project research

Research completed September 17, 2026. Infrastructure/backend priority; CV/ML included; no coding-time cutoff.

Selected project: **#3, Coverage-Aware Dataset Compiler**. The subsequent [build-preparation plan](build_plan/README.md) contains the Roboflow-first/Modal hybrid architecture, account checklist, source-level trainer design, dataset protocol, reuse map and independent review. That plan is design work; implementation and training have not started.

- [Interactive explorer](ideas.html): filter 12 surviving ideas and compare up to three. Standalone HTML; no server or network needed.
- [Full report](REPORT.md): company analysis, the shortlist, all proposal details and recommended proof milestones.
- [Selection audit](AUDIT.md): 44 named lane candidates, merges/rejections, evidence limits and workflow.
- [Novelty critique](critiques/novelty.md) and [technical critique](critiques/technical.md): independent adversarial reviews.
- [Structured idea cards](ideas.json), [search log](search_log.json) and [research counts](research_stats.json).

The seven detailed investigations are [company](lanes/company.md), [inference/edge](lanes/inference_edge.md), [data](lanes/data.md), [CV/models](lanes/cv_models.md), [Workflows/developer experience](lanes/workflows_devex.md), [customers](lanes/customers.md), and [infrastructure/security](lanes/infra_security.md).

`raw/` retains Exa results, primary-source snapshots and the bounded synthetic DrawerDissect research probe. Search results are not all verified evidence; use the lane reports for supporting claims and confidence. Source snapshots are research material, not new project implementation.

The Exa helper reads the credential location referenced by the user's `dbresearch` setup. No keys were copied into this workspace or the report. No outreach, issue filing, PR submission, purchases or Roboflow production changes were made.

Regenerate the report and explorer after editing the cards/templates:

```sh
python3 research/roboflow/tools/build_deliverables.py
```
