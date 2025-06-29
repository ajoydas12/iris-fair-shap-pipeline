# MLops Course Week 2: Data & Model Versioning with DVC

This project demonstrates data and model versioning using DVC (Data Version Control) and Git, following best practices for reproducible machine learning workflows.

## Project Structure

- `main.py` — Main script for training and saving the model.
- `runner.sh` — Automates DVC and Git versioning for your own dataset and model.
- `data/iris.csv` — Example dataset (Iris dataset).
- `requirements.txt` — Python dependencies.

## Getting Started

### Data & Model Versioning (`runner.sh`)
This script:
- Initializes Git and DVC if needed
- Ensures `data/iris.csv` is tracked by DVC, not Git
- Sets up a Python virtual environment and installs dependencies
- Tracks data and model versions with DVC and Git tags
- Demonstrates updating the dataset and retraining the model

**Usage:**
```bash
bash runner.sh
```

## Requirements
- Python 3.7+
- [DVC](https://dvc.org/doc/install)
- Git

## Key Concepts
- **Data Versioning:** DVC tracks large data files outside Git, enabling reproducible experiments.
- **Model Versioning:** Model artifacts are versioned with DVC and Git tags.
- **Reproducibility:** You can reproduce any experiment state using Git and DVC checkout.

## References
- [DVC Documentation](https://dvc.org/doc)

---

Feel free to modify the scripts for your own datasets and workflows!
