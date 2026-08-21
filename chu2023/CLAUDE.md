# Claude Code project instructions

## Project and authority

This is a scientific-computing subproject for implementing and validating the
Chu et al. pair approximation and a finite-population agent-based model (ABM)
of multi-agent Q-learning.

Before work, read the documents relevant to the task:

- `AGENTS.md` — binding local safety, scope, and validation rules.
- `MODEL_SPEC.md` — authoritative scientific conventions and equations.
- `PLAN.md` — implemented milestones, exit gates, and deferred work.
- `README.md` — current commands and user-facing implementation summary.
- `docs/LAMBDA_GPU_PILOT.md` — detailed GPU-pilot operator runbook.

Do not treat this file as a substitute for those documents. If instructions
conflict, stop and ask the human; never silently change the scientific model.

## Working rules

- Work only inside this `chu2023` subproject. Inspect Git status and the
  relevant documentation before changing anything.
- Keep work narrow and evidence-backed. Make the smallest justified change;
  do not perform speculative rewrites or unrelated cleanup.
- Preserve `case2_1.py` and `pair-approx_multi-agent_stochastic_games.pdf`
  byte-for-byte. Do not restore intentionally deleted legacy files.
- Keep scientific claims separate from operational or performance claims.
  Implemented code is not proof that a particular hardware path was exercised.
- Run focused validation appropriate to the change and report the exact
  commands, results, skips, and remaining uncertainty before claiming success.
- Preserve reproducibility artifacts and fail-closed safeguards. Do not weaken
  configuration, resource, executable-identity, capacity, or science gates.
- Never request, expose, paste, or store SSH private keys, tokens, passwords,
  cloud credentials, private URLs, or other secrets in repository files, chat,
  commands, logs, configurations, or metadata.

Unless the human explicitly authorizes the specific action, do not:

- commit, push, create branches, or alter repository history;
- contact cloud services or other external systems;
- provision, stop, terminate, or otherwise operate cloud resources;
- install packages or modify system Python, drivers, or system CUDA;
- make large allocations or perform full-grid work; or
- lower, compile, allocate, or execute a GPU pilot stage.

## Immediate Lambda boundary

The current operator-provided context is that one Lambda instance is live and
the repository has been cloned there. This is external context, not a
repository-validated GPU result. The next authorized task is only to preview
and create the project-local environment exactly as documented, run the GPU
doctor, and report its result. Stop there. Any small, medium, large, analysis,
or execution stage requires subsequent explicit human review and approval.

For `G=131`, do not allocate, lower, compile, or execute anything except through
the explicit prerequisite and confirmation protocol in
`docs/LAMBDA_GPU_PILOT.md`, and only after the human authorizes that exact stage.

Stop and report rather than proceeding whenever documentation is ambiguous or
any scientific-scope, resource-admission, executable-identity, environment,
device-identity, or capacity check fails or is unavailable.
