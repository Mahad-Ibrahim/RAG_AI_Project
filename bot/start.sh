#!/usr/bin/env bash

python ingest.py

uvicorn main:app --host 0.0.0.0 --port 8000

