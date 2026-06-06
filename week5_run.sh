#!/usr/bin/env bash
# ============================================================
# ComorbidAlert — Week 5 Runner
# Run from your project root with the .venv activated:
#   cd ~/comorbid_alret
#   source .venv/bin/activate
#   bash week5_run.sh 2>&1 | tee logs/week5_$(date +%Y%m%d_%H%M).log
# ============================================================

set -e

echo "============================================================"
echo " ComorbidAlert — Week 5"
echo " $(date)"
echo "============================================================"

echo ""
echo "▶  Step 1/3 — Weighted Ensemble"
python forecasting/week5_01_ensemble.py

echo ""
echo "▶  Step 2/3 — Alerting System"
python forecasting/week5_02_alerting.py

echo ""
echo "▶  Step 3/3 — Validation & Visualisation"
python forecasting/week5_03_visualise.py

echo ""
echo "============================================================"
echo " Week 5 Complete "
echo " Check S3 outputs:"
echo "   aws s3 ls s3://comorbid-alert-data/alerts/"
echo "   aws s3 ls s3://comorbid-alert-data/outputs/ | grep week5"
echo "============================================================"