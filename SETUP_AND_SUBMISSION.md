# Lab 07: Setup, validation and submission walkthrough

## Before you begin

- Install Anaconda/Miniconda, Git, and VS Code (Python extension).
- Extract the ZIP to a **new** folder named `lab07-first-principles`.
- Open that folder in VS Code using **File > Open Folder**.
- Open **Terminal > New Terminal** (Windows PowerShell or Anaconda Prompt).
- Do not submit the ZIP as a replacement for the required GitHub repository URL.

## 1. Important: the first Git commit

The assignment specifically requires `bikes_data.py` in the **first commit by
itself**, followed by at least four further meaningful commits. This ZIP is
provided **without a fake Git history**. In the folder, before making other
commits:

```bash
git init
git branch -M main
git add bikes_data.py
git commit -m "Add supplied seeded bike rental data generator"
```

Then stage and commit the remaining files in meaningful groups, for example:

```bash
git add prepare.py model.py environment.yml requirements.txt requirements-dev.txt pyproject.toml .gitignore
git commit -m "Add leak-free preparation, network and environment"
git add train.py
git commit -m "Implement manual epoch loop and controlled experiments"
git add tests/
git commit -m "Add split, model and learning loop tests"
git add reports/ README.md SETUP_AND_SUBMISSION.md
git commit -m "Document results and commit reproducible reports"
```

If you edit files during the process, commit the changes with another
meaningful message rather than inventing a history. To see the commits:

```bash
git log --oneline
```

## 2. Install the environment

Inside the project folder:

```bash
conda env create -f environment.yml
conda activate lab07-first-principles
python --version
python -m pip list
```

The lab specifically requires Python 3.12 and conda-forge; PyTorch is
installed through pip. Package versions are pinned in the requirements files.
If your platform cannot install the specified pinned versions, consult the
course instructions and use `python -m pip freeze` to record the exact
compatible versions you actually installed (do not silently omit pins).

## 3. Inspect given data and run everything

```bash
python bikes_data.py
python train.py
```

This creates or overwrites all expected artifacts inside `reports/`:

- `learning_curves.png` and `rentals_by_hour.png`
- `curves_low_learning_rate.png` and `curves_wide_network.png`
- `history.csv` plus one `history_<run>.csv` per run
- `experiments.csv` and `metrics.json`

The split is train/validation/test = 3600/1200/1200 and the dataset has
6000 rows. Model choice is based only on validation loss, followed by a
single test evaluation. Compare the metrics with the README; numerical
rounding can differ slightly on other computers.

## 4. Mandatory checks

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

If the formatting check fails:

```bash
python -m ruff format .
python -m ruff check --fix .
python -m ruff format --check .
python -m ruff check .
python -m pytest
python train.py
```

Review automated changes. Use `python -m`, not bare `pytest` or `ruff`.

## 5. Push to GitHub

Create a new, empty GitHub repository named `lab07-first-principles`
(do not initialize it with another README). Then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/lab07-first-principles.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Verify `reports/` is in
the GitHub repository, including all PNG/CSV/JSON results.

## 6. Fresh clone test (mandatory)

In a different parent folder, clone the repository and run:

```bash
git clone https://github.com/YOUR_USERNAME/lab07-first-principles.git
cd lab07-first-principles
conda env create -f environment.yml
conda activate lab07-first-principles
python -m ruff format --check .
python -m ruff check .
python -m pytest
python train.py
```

The instructor requires a public repository or a private repository shared
with the specified instructor account. Submit the **repository link** on
Blackboard by the deadline shown there, not an invented or unverified link.

## Common problems

- `ModuleNotFoundError`: activate the Conda environment in the correct folder.
- `conda` not found: open Anaconda Prompt or initialize Conda in your shell.
- `ruff` not installed: rerun `conda env create ...` or install
  `requirements-dev.txt` inside the environment.
- `tests` cannot import files: run commands from the repository root.
- Can't see generated figures: check that `reports/` wasn't mistakenly ignored.
- Model seems to learn nothing: inspect `optimizer.zero_grad()`,
  `loss.backward()`, `optimizer.step()`, and prediction/target `(N,1)` shapes.
- Numbers change: verify deterministic seeds and the exact installed versions.
- Original assignment PDF indentation: the included given `bikes_data.py`
  reproduces the supplied statements, with PDF indentation normalized for a
  working Python file. Do not change its data generation algorithm.
