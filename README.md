# AutoMLBench
The aim of this project is to simplify benchmarking process in the field of automated machine learning.

### Project status
The project under active development and new frameworks, tasks, metrics and repositories to be added soon.
- Supported frameworks: 
    - [AutoGluon.Tabular](https://github.com/autogluon/autogluon).
    - [Imbaml](https://github.com/AxiomAlive/Imbaml).
- Supported tasks: tabular classification. 
- Supported metrics:
    - F1.
    - Precision.
    - Recall.
    - ROC AUC.
    - Average precision.
    - Balanced accuracy.
    - MCC.
    - Accuracy. 
- Supported data repositories:
    - [OpenML](https://www.openml.org/).
    - Local filesystem.

### Usage
1. Clone the project.
2. Initialize project with `uv init` and create a virtual environment with `uv venv -p 3.10`.
3. Install dependencies with `uv sync`.
4. Use examples from **examples** folder as a starting point. For instance, `uv run -m examples.1`.

### Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.

This is my personal project and it has no funding at all.
