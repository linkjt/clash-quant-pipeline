#!/bin/bash
cd /mnt/typeshyt/thehub/python

# Verify python exists in env
if [ ! -f env/bin/python ]; then
    echo "Error: env/bin/python not found. Please recreate the virtual environment first."
    exit 1
fi

if pgrep "streamlit" > /dev/null; then
    echo "Killing Streamlit"
    pkill "streamlit"
fi
echo "Running daily quantitative signal generation..."

. env/bin/activate
env/bin/python LSTM.py
env/bin/python train_xgb.py
env/bin/python train_deck2vec.py
env/bin/python quant.py
env/bin/python alpha.py
streamlit run datavis.py --server.port 9021

echo "Generating extrapolated deck recommendation..."
#env/bin/python deck_builder.py

echo "Daily pipeline execution complete."
