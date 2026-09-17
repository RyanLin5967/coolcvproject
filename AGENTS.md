# Project working instructions

## Restart entrypoint

Read [HANDOFF.md](HANDOFF.md) first, then [the build-plan index](research/roboflow/build_plan/README.md). Planning/research are complete. No application code, dependency environment, authenticated provider setup, training or deployment has been executed. Resume from the recorded state; do not restart company research or silently treat planned tests as passed.

The user wants autonomous work without repeated permission prompts for authorized actions. The active session is configured for full filesystem/network access and approval policy `never`. Follow the actual current system/developer permission instructions; do not invent approval requests for routine reads, research, edits, or checks.

On 2026-09-17, research agents created before the permission change retained `workspace-write` instructions and generated unnecessary approval prompts when reused. Those old agents were stopped. Do not resume `/root/data`, `/root/cv_models`, `/root/inference_edge`, `/root/attack_technical`, or `/root/attack_novelty` for tool work in this session. Their findings can be reused. When delegation is authorized, create a fresh agent from the current parent and check that its effective permission instructions match before tool work. Under approval policy `never`, omit `sandbox_permissions` and do not request escalation.

Current project stage: planning complete; prepared for the user's subsequent implementation/testing phase. Latest requirement: Roboflow must be used, preferably for hosted training; a Roboflow/Modal hybrid is acceptable. The restart request authorizes preserving state, not launching training. Keep credentials out of chat, committed files, reports, and tool output. Use the existing Exa helper for research credentials.
