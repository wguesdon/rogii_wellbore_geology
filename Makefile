PY := uv run python

.PHONY: help run-multipath run-single-path harness-status track-list

help:
	@printf '%s\n' 'make run-multipath   Run the final multipath solution.'
	@printf '%s\n' 'make run-single-path Run the final single path solution.'
	@printf '%s\n' 'make harness-status  Inspect the model search status.'
	@printf '%s\n' 'make track-list      List locally tracked Kaggle runs.'

run-multipath:
	$(PY) solutions/multipath/solution.py

run-single-path:
	$(PY) solutions/single_path/solution.py

harness-status:
	$(PY) harness/breadth_gate.py --best-cv 7.1593

track-list:
	$(PY) tracking/track_kaggle_runs.py list
