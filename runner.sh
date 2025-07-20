#!/bin/bash

# Automated ML project setup using Git and DVC
# Author: Ajoy Das

# Step 0: Configure Git globally (only needs to be done once per system)
git config --global user.email "ajoyd0957@gmail.com"
git config --global user.name "ajoydas12"

echo "========== [0] Checking Git Initialization =========="
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Creating a new Git repository..."
    git init
    git add .
    git commit -m "Initial commit setup"
else
    echo "Git repo already exists"
fi

echo "========== [1] Removing iris.csv from Git =========="
if git ls-files --error-unmatch data/iris.csv &> /dev/null; then
    echo "Untracking iris.csv from Git (handing off to DVC)..."
    git rm --cached data/iris.csv
fi

echo "========== [2] Setting Up Python Environment =========="
python3 -m venv .venv
source .venv/bin/activate

echo "========== [3] Installing Python Dependencies =========="
pip install --upgrade pip
pip install -r requirements.txt
pip install dvc

echo "========== [4] DVC Initialization =========="
if [ ! -d ".dvc" ]; then
    dvc init
    git add .dvc .dvc/.gitignore
    git commit -m "Initialize DVC version control"
else
    echo "DVC already set up"
fi

echo "========== [5] Versioning Dataset - V1 =========="
dvc add data/iris.csv
git add data/iris.csv.dvc .gitignore
git commit -m "Track iris.csv as Version 1"

echo "========== [6] Model Training - V1 =========="
python main.py
echo "Saving trained model (V1)..."
mkdir -p models
mv models/decision_tree_model.pkl models/decision_tree_model_v1.pkl
git add models/decision_tree_model_v1.pkl
git commit -m "Store trained model as V1"
git tag V1

echo "========== [7] Creating Dataset V2 =========="
head -n 100 data/iris.csv > data/temp.csv && mv data/temp.csv data/iris.csv
dvc add data/iris.csv
git add data/iris.csv.dvc
git commit -m "Updated dataset: Version 2 (100 rows only)"

echo "========== [8] Model Training - V2 =========="
python main.py
echo "Saving trained model (V2)..."
mv models/decision_tree_model.pkl models/decision_tree_model_v2.pkl
git add models/decision_tree_model_v2.pkl
git commit -m "Store trained model as V2"
git tag V2

echo "========== [9] Version Comparison =========="

echo "Comparing data file sizes:"
git checkout V1
dvc checkout
dvc pull
echo "➡️ V1 data line count:"
wc -l data/iris.csv

git checkout V2
dvc checkout
dvc pull
echo "➡️ V2 data line count:"
wc -l data/iris.csv

echo "Comparing model file checksums:"
git checkout V1
dvc checkout
dvc pull
echo "✅ Model checksum (V1):"
(md5sum models/decision_tree_model_v1.pkl 2>/dev/null) || (shasum models/decision_tree_model_v1.pkl)

git checkout V2
dvc checkout
dvc pull
echo "✅ Model checksum (V2):"
(md5sum models/decision_tree_model_v2.pkl 2>/dev/null) || (shasum models/decision_tree_model_v2.pkl)

echo "=========== ✅ Pipeline Finished Successfully ==========="
