PY := uv run python

.PHONY: help preflight preflight-quick gate gate-list ensemble breadth-status \
	lock-check state claude-handoff run-multipath run-single-path track-list \
	harness-status

help:
	@printf '%s\n' 'make preflight        Verify artifact layout and the known winner records.'
	@printf '%s\n' 'make preflight-quick  Verify the environment and artifact layout only.'
	@printf '%s\n' 'make gate BASE=name   Gate a candidate and its rolled null.'
	@printf '%s\n' 'make ensemble         Run reciprocal split ensemble selection.'
	@printf '%s\n' 'make breadth-status   Inspect the model search status.'
	@printf '%s\n' 'make claude-handoff   Write a continuation brief.'
	@printf '%s\n' 'make run-multipath    Run the final multipath inference pipeline.'
	@printf '%s\n' 'make run-single-path  Run the final single path inference pipeline.'
	@printf '%s\n' 'make track-list       List locally tracked Kaggle runs.'

preflight:
	$(PY) harness/release_preflight.py

preflight-quick:
	$(PY) harness/release_preflight.py --quick

gate:
	@test -n "$(BASE)" || { echo "usage: make gate BASE=<name>"; exit 2; }
	$(PY) harness/gate.py --base $(BASE)

gate-list:
	$(PY) harness/gate.py --list

ensemble:
	$(PY) harness/ensemble.py

breadth-status harness-status:
	$(PY) harness/breadth_gate.py --best-cv $(or $(BEST_CV),7.1593)

lock-check:
	$(PY) harness/breadth_gate.py --best-cv $(or $(BEST_CV),7.1593) --lock $(if $(OVERRIDE),--override "$(OVERRIDE)",)

state:
	@echo "=== LEDGER.tsv (last 8) ==="; tail -8 LEDGER.tsv | cut -c1-150
	@echo ""; echo "=== QUEUE.md item 1 ==="; sed -n '/^## 1\./,/^## 2\./p' QUEUE.md | head -20
	@echo ""; echo "=== forward model breadth ==="; $(PY) harness/breadth_gate.py --best-cv 7.1593
	@echo ""; echo "=== recent commits ==="; git log --oneline -12 -- .

claude-handoff:
	$(PY) harness/claude_handoff.py --best-cv $(or $(BEST_CV),7.1593)

run-multipath:
	$(PY) solutions/multipath/solution.py

run-single-path:
	$(PY) solutions/single_path/solution.py

track-list:
	$(PY) tracking/track_kaggle_runs.py list
