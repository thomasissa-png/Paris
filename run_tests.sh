#!/usr/bin/env bash
# Run all tests — execute this before every commit
set -e
echo "=============================="
echo " Polymarket Scanner — Tests"
echo "=============================="
python3 -m pytest tests/ -v --tb=short
echo ""
echo "All tests passed."
