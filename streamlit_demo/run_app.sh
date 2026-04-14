#!/usr/bin/env bash
set -euo pipefail

# This small wrapper lets you start the app without remembering the full
# Streamlit command. Run it from the project root with:
#
#   bash streamlit_demo/run_app.sh

streamlit run streamlit_demo/app.py \
    --client.toolbarMode minimal \
    --browser.gatherUsageStats false
