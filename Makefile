.PHONY: build test fmt clippy download train export-gguf help

CARGO ?= cargo
INSTALL_DIR ?= $(HOME)/.local/bin

all: llm-burner

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
	@echo "Building release."
	$(CARGO) build --release
	@echo "Release ready."

build-debug:
	@echo "Building debug."
	$(CARGO) build
	@echo "Debug build ready."

test:
	@echo "Running unit tests."
	$(CARGO) test
	@echo "Test done."

fmt:
	@echo "Formatting."
	$(CARGO) fmt
	@echo "Formatting done."

clippy:
	@echo "Running Clippy."
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	@echo "Clippy done."

download:
	@echo "--- Start downloading ---"
	$(CARGO) run -- download $(ARGS)
	@echo "--- End Downloading"

train:
	@echo "--- Start training ---"
	$(CARGO) run --release -- train $(ARGS)
	@echo "--- End training ---"

export-gguf:
	@echo "Exporting to GGUF."
	$(CARGO) run --release -- export $(ARGS)
	@echo "Export done."

clean:
	@echo "Removing `target` folder."
	rm -rf target
	@echo "`target` folder deleted."

install:
	@echo "Moving binary to $(INSTALL_DIR)."
	mv target/release/llm-burner $(INSTALL_DIR)
	@echo "Install done."

uninstall:
	@echo "Removing binary from $(INSTALL_DIR)."
	rm -f $(INSTALL_DIR)/llm-burner
	@echo "Deleted binary."

llm-burner: uninstall clean build install clean
