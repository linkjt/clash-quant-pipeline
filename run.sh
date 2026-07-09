#!/bin/bash
export PYTHONUNBUFFERED=1
cd /mnt/typeshyt/thehub/python
# activation not required when running via env/bin/python directly

env/bin/pip install --upgrade pip wheel setuptools
env/bin/pip install --only-binary=:all: -r requirements.txt
echo "Purging lingering backend database connections..."


# Safe fallback initialization
#python fix.py

echo "Starting synchronized pipeline..."
# Launch scraper safely to background
env/bin/python grabmatches.py &

# Launch processing core in foreground to stream logs safely
env/bin/python formatmatches.py