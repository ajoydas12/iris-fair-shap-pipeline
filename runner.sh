
#!/bin/bash

# This script automates the setup and versioning of a machine learning project using DVC and Git.
git config --global user.email "ajoyd0957@gmail.com"
git config --global user.name "ajoydas12"

echo "********************"
echo "Step 0: Initialize Git repository if needed"
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Initializing new Git repository..."
    git init
    git add .
    git commit -m "Initial commit"
else
    echo "Git repository already initialized"
fi

echo "********************"
echo "Step 1: Ensure data/iris.csv is not tracked by Git"
if git ls-files --error-unmatch data/iris.csv > /dev/null 2>&1; then
    echo "Removing data/iris.csv from Git history (index)..."
    git rm --cached data/iris.csv
fi

echo "******************************"
echo "Step 2: Create and activate virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "******************************"
echo "Step 3: Install requirements and DVC"
pip install --upgrade pip
pip install -r requirements.txt
pip install dvc

echo "******************************"
echo "Step 4: Initialize DVC"
if [ ! -d ".dvc" ]; then
    dvc init
    git add .dvc .dvc/.gitignore
    git commit -m "Initialize DVC"
else
    echo "DVC already initialized"
fi

echo "******************************"
echo "Step 5: Track dataset (V1)"
dvc add data/iris.csv
git add data/iris.csv.dvc .dvc .gitignore
git commit -m "Add iris.csv as V1"

echo "******************************"
echo "Step 6: Train model and save it (V1)"
python main.py

echo "=> Saving model as pickle (V1)"
mkdir -p models
mv models/decision_tree_model.pkl models/decision_tree_model_v1.pkl
git add models/decision_tree_model_v1.pkl
git commit -m "Train and save model for V1"
git tag V1

echo "******************************"
echo "Step 7: Modify dataset to create V2"
head -n 100 data/iris.csv > data/iris_v2.csv
mv data/iris_v2.csv data/iris.csv

echo "=> Re-track modified data"
dvc add data/iris.csv
git add data/iris.csv.dvc
git commit -m "Update iris.csv to V2 (new 100 rows)"

echo "******************************"
echo "Step 8: Train model and save it (V2)"
python main.py

echo "=> Saving model as pickle (V2)"
mv models/decision_tree_model.pkl models/decision_tree_model_v2.pkl
git add models/decision_tree_model_v2.pkl
git commit -m "Train and save model for V2"
git tag V2

echo "**************************************"
echo "Step 9: Compare data and model versions"
echo "**************************************"

# Compare data versions
echo "******************************"
echo "📊 Data file sizes for each version:"

echo "🔁 Checking out V1 data..."
git checkout V1
dvc checkout
dvc pull
echo "V1 Data Size:"
wc -l data/iris.csv

echo "🔁 Checking out V2 data..."
git checkout V2
dvc checkout
dvc pull
echo "V2 Data Size:"
wc -l data/iris.csv

# Compare model versions
echo "*****************************************"
echo "🧠 Model file checksums for each version:"
echo "*****************************************"

git checkout V1
dvc checkout
dvc pull
echo "V1 Model Checksum:"
md5sum models/decision_tree_model_v1.pkl || shasum models/decision_tree_model_v1.pkl

git checkout V2
dvc checkout
dvc pull
echo "V2 Model Checksum:"
md5sum models/decision_tree_model_v2.pkl || shasum models/decision_tree_model_v2.pkl

echo "**************************************"
echo "All steps completed successfully!"
echo "***************************************"
