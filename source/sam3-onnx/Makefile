ifneq ($(OS),Windows_NT)
  # On Unix-based systems, use ANSI codes
  BLUE = \033[36m
  BOLD_BLUE = \033[1;36m
  RED = \033[31m
  YELLOW = \033[33m
  BOLD = \033[1m
  NC = \033[0m
endif

escape = $(subst $$,\$$,$(subst ",\",$(subst ',\',$(1))))

define exec
	@echo "$(BOLD_BLUE)$(call escape,$(1))$(NC)"
	@$(1)
endef

help:
	@echo "$(BLUE)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-].+:.*?# .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?# "}; \
		{printf "  $(YELLOW)%-23s$(NC) %s\n", $$1, $$2}'

uv:
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(BOLD)uv is already installed.$(NC)"; \
	else \
		echo "$(RED)Please install uv https://docs.astral.sh/uv/getting-started/installation/$(NC)"; \
	fi

build: uv  # build
	$(call exec,git submodule update --init --recursive)
	$(call exec,uv sync)

lint:  # lint
	$(call exec,uv run ruff format --check)
	$(call exec,uv run ruff check)
	$(call exec,uv run ty check)

format:  # format
	$(call exec,uv run ruff format)
	$(call exec,uv run ruff check --fix)
