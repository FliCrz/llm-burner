.PHONY: build test fmt clippy download train export-gguf help

CARGO ?= cargo

help:
	@echo "Available targets:"
	@echo "  make build            Build the project"
	@echo "  make test             Run tests"
	@echo "  make fmt              Format Rust sources"
	@echo "  make clippy           Run clippy"
	@echo "  make download         Run the download command"
	@echo "  make download ARGS='...' Run the download command with extra flags"
	@echo "  make train            Run the train command"
	@echo "  make train ARGS='...' Run the train command with extra flags"
	@echo "  make export-gguf      Run the export command"
	@echo "  make export-gguf ARGS='--model-dir DIR --output FILE [--model-name NAME]'"
	@echo "                        DIR must contain config.json, tokenizer.json,"
	@echo "                        and the fine-tuned .safetensors"

build:
	$(CARGO) build

test:
	$(CARGO) test

fmt:
	$(CARGO) fmt

clippy:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

download:
	$(CARGO) run -- download $(ARGS)

train:
	$(CARGO) run --release -- train $(ARGS)

export-gguf:
	$(CARGO) run --release -- export $(ARGS)
