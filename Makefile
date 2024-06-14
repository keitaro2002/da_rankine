# Makefile

# 変数の定義
SIF_PATH := /data10/imageshare/kinuki/cuda116_py39_jupyter.sif
VENV_PATH := /data10/kinuki/da_rankine/.venv
JUPYTER_CMD := source $(VENV_PATH)/bin/activate && poetry run jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root
PYTHON_SCRIPT := src/models/LETKF.py  # 実行するPythonスクリプトのパスを指定
LOG_DIR := logs
INTERVAL := 50

# ターゲットの定義

.PHONY: activate_jupyter
activate_jupyter:
	@if [ "$(env)" = "cpu" ]; then \
		srun --chdir $$(pwd) singularity exec $(SIF_PATH) /bin/bash -c "$(JUPYTER_CMD)"; \
	elif [ "$(env)" = "gpu" ]; then \
		srun --chdir $$(pwd) --gpus 40gb:1 -p gpu singularity exec --nv $(SIF_PATH) /bin/bash -c "$(JUPYTER_CMD)"; \
	else \
		echo "Invalid argument: $(env). Use 'cpu' or 'gpu'."; \
		exit 1; \
	fi

.PHONY: add_module_poetry
add_module_poetry:
	@if [ -z "$(module)" ]; then \
		echo "Usage: make add_module_poetry module=<module_name>"; \
		exit 1; \
	else \
		PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring poetry add $(module); \
	fi

.PHONY: run_parallel
run_parallel:
	mkdir -p $(LOG_DIR)
	@for (( i=$(start); i<$(end); i+=$(INTERVAL) )); do \
	  P_RANGE_START=$$i; \
	  P_RANGE_END=$$((i+$(INTERVAL)-1)); \
	  P_RANGE_END=$$((P_RANGE_END<$(end)?P_RANGE_END:$(end)-1)); \
	  LOG_FILE="$(LOG_DIR)/log_$${P_RANGE_START}_$${P_RANGE_END}.txt"; \
	  srun --chdir $$(pwd) -p cpu singularity run $(SIF_PATH) /bin/bash -c "source $(VENV_PATH)/bin/activate && poetry run python $(PYTHON_SCRIPT) $$P_RANGE_START $$P_RANGE_END" > $$LOG_FILE 2>&1 & \
	done
	wait

.PHONY: run
run:
	@if [ "$(env)" = "gpu" ]; then \
		srun --chdir $$(pwd) --gpus 20gb:1 -p gpu singularity run --nv $(SIF_PATH) /bin/bash -c "source $(VENV_PATH)/bin/activate && poetry run python $(script) $(args)"; \
	elif [ "$(env)" = "cpu" ]; then \
		srun --chdir $$(pwd) -p cpu singularity run $(SIF_PATH) /bin/bash -c "source $(VENV_PATH)/bin/activate && poetry run python $(script) $(args)"; \
	else \
		echo "指定された実行環境が不正です。gpuまたはcpuを指定してください。"; \
	fi

# lintをかける
.PHONY: lint
lint:
	poetry run pysen run lint

# format違反をある程度自動修正する
.PHONY: format
format:
	poetry run pysen run format

.PHONY: test
test:
	poetry run pytest -s -vv ./tests

# 変数を指定
env := cpu
module := numpy # 追加するモジュール名
start := 0
end := $$(echo $((91 * 91))) # end 変数の計算を修正
script := src/data/dataset.py # 実行するPythonスクリプトのパスを指定
args := some_args # スクリプトに渡す引数を指定


# make activate_jupyter
# make add_module_poetry
# make run_parallel
# make run
