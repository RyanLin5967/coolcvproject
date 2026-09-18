# Project working instructions

## Restart entrypoint

Read [EXECUTION_STATUS.md](EXECUTION_STATUS.md) first, then [HANDOFF.md](HANDOFF.md) for historical context and [the build-plan index](research/roboflow/build_plan/README.md). Implementation is now authorized; the execution record supersedes planning-only statements in older documents. Resume from recorded evidence; do not restart company research or silently treat planned tests as passed.

The user wants autonomous work without repeated permission prompts for authorized actions. The active session is configured for full filesystem/network access and approval policy `never`. Follow the actual current system/developer permission instructions; do not invent approval requests for routine reads, research, edits, or checks.

On 2026-09-17, research agents created before the permission change retained `workspace-write` instructions and generated unnecessary approval prompts when reused. Those old agents were stopped. Do not resume `/root/data`, `/root/cv_models`, `/root/inference_edge`, `/root/attack_technical`, or `/root/attack_novelty` for tool work in this session. Their findings can be reused. When delegation is authorized, create a fresh agent from the current parent and check that its effective permission instructions match before tool work. Under approval policy `never`, omit `sandbox_permissions` and do not request escalation.

Current project stage: autonomous implementation/testing authorized, including provider integration within existing credits. Roboflow must be used, preferably for hosted training; a Roboflow/Modal hybrid is acceptable. Hard user budget: $0 out of pocket. Verify provider controls before cloud execution. Keep credentials outside the repository, artifacts and tool output. Use the existing Exa helper for research credentials.

On 2026-09-18 the user explicitly prohibited local ML compute because training froze their computer. Run training and model evaluation on cloud GPUs within the confirmed credit budget. Do not restart MPS/CPU experiments locally. Lightweight code edits, metadata checks and artifact transfers may remain local.
