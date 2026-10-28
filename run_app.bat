@echo off
title Mobile Phone Customer Churn Classifier
cd /d "%~dp0"
echo ===================================================
echo Starting Customer Churn Prediction Web Application...
echo ===================================================
streamlit run app.py
pause
