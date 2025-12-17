#!/bin/bash
# Wrapper script to run decrypt_wallet.py with proper environment
cd "$(dirname "$0")"
source venv/bin/activate
python decrypt_wallet.py
