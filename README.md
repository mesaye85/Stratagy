# Stratagy Dataset Processing

This repository contains a CSV file with several economic and political indicators for different countries and a Python script that processes that data.

## Dataset

The file `main.csv` provides multiple metrics per country:

- **Country** – used for the Gini index and GDP values.
- **Gini** – the Gini coefficient.
- **GDP** – gross domestic product (in unspecified units).
- **Country_military** – country name associated with the military metric.
- **Military** – numeric score for military strength.
- **Country, Relations** – combined column with a country name and a relations score.
- **Country_democracy** – country name associated with the democracy metric.
- **Democracy** – numeric measure of democracy.

The `main.py` script splits and renames these columns, cleans country names, merges all metrics by country and outputs a cleaned file called `main_merged.csv`. It also generates a plot of Gini coefficients.

## Prerequisites

- Python 3.10+
- The dependencies listed in `pyproject.toml` (pandas, matplotlib, seaborn, scikit-learn)

You can install them using `poetry` or plain `pip` as described below.

### Using Poetry

1. Install Poetry if you don’t have it:
   ```bash
   pip install poetry
   ```
2. Install the project dependencies:
   ```bash
   poetry install
   ```
3. Run the script:
   ```bash
   poetry run python main.py
   ```

### Using pip and a virtual environment

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install the required packages:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
3. Execute the script:
   ```bash
   python main.py
   ```

After running the script you should find `main_merged.csv` in the repository directory along with a `gini_coefficients.png` image.
