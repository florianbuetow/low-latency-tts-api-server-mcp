#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="./data/models"
MODEL_NAME="Kokoro_no_espeak.gguf"
MODEL_PATH="$BASE_DIR/$MODEL_NAME"
MODEL_URL="https://huggingface.co/mmwillet2/Kokoro_GGUF/resolve/main/$MODEL_NAME"

printf "\033[34mAvailable Kokoro TTS models:\033[0m\n\n"
if [ -f "$MODEL_PATH" ]; then
    printf "  1. %s   \033[32m[downloaded]\033[0m\n" "$MODEL_NAME"
else
    printf "  1. %s\n" "$MODEL_NAME"
fi
printf "\n"

read -rp "Select model [1]: " choice

case "$choice" in
    "" | 1) ;;
    *)
        printf "\033[31mInvalid choice.\033[0m\n"
        exit 1
        ;;
esac

mkdir -p "$BASE_DIR"

if [ -f "$MODEL_PATH" ]; then
    printf "\n\033[32mAlready downloaded: %s\033[0m\n" "$MODEL_PATH"
    exit 0
fi

printf "\n\033[34mDownloading %s to %s\033[0m\n\n" "$MODEL_NAME" "$MODEL_PATH"
curl -L --fail -C - --progress-bar -o "$MODEL_PATH" "$MODEL_URL"

printf "\n\033[32mDone. Model saved to %s\033[0m\n" "$MODEL_PATH"
printf "\nUpdate config.yaml model to:\n  model: %s\n" "$MODEL_PATH"
