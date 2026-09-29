# srm/legacy: frozen reference code

Verified scaffold from before the monorepo migration (5x scale, 51->256 replicate-pad hack).
Rules:
1. Read-only reference. Fixes go into the new modules, never here.
2. New code must not `import srm.legacy...`; port the logic and add tests.
3. Legacy modules use flat imports (`from models_v3 import ...`), so run them from this folder:
   `cd srm/legacy && python main.py`.
4. Ruff and CI skip this folder.
5. Known defects are tracked in `srm/docs/DEFECT_AUDIT.md`.
The old verification log is kept in `README_legacy_verification_log.md`.
