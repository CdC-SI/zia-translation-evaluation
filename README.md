# Zia Translation Evaluation

A pipeline for evaluating the quality of the Zia document translation
service. It sends documents (PDFs / images) through the translation service,
scores the resulting translations against the originals using an
LLM-as-a-judge (MQM methodology), and aggregates the results into
human-readable Markdown reports.

The pipeline has three stages, each runnable independently or chained
together with `run_pipeline.py`:

1. **Translation** (`src/pipeline/translation.py`) – sends documents to the
   Zia translation service and saves the translated Markdown/PDF output.
2. **Evaluation** (`src/pipeline/evaluation.py`) – compares each translation
   against its original document with a vision/language model judge and
   produces a scored JSON result.
3. **Reporting** (`src/pipeline/report.py`) – turns one or more evaluation
   JSON results into a consolidated Markdown report.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) to manage its Python
environment and dependencies.

If `uv` is not installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project root, install the dependencies:

```bash
uv sync
# if CA bundle needed
SSL_CERT_FILE=/path/to/ca-certificates.crt uv sync
```
Note that you might need 

Copy `.env.example` to `.env` and adjust the values (translation service URL,
LLM judge endpoint/credentials, etc.):

```bash
cp .env.example .env
```

> The Zia translation service must be running and reachable before running
> the translation step (see that service's own setup instructions).

## Commands

Every command below shows only its minimum required parameters. Pass
`--help` to any script for the full list of options, and see
[`docs/command_examples.md`](docs/command_examples.md) for more examples.

### Run the full pipeline (translate + evaluate + report)

```bash
uv run src/pipeline/run_pipeline.py --input <file-or-folder> --target-language <lang> --strategy <single|dual> --output <existing-folder>
```

### Translate documents

```bash
uv run src/pipeline/translation.py --input <file-or-folder> --target-language <lang> --strategy <single|dual>
```

### Evaluate translations

```bash
uv run src/pipeline/evaluation.py --original <file-or-folder> --translated <file-or-folder> --result-dir <folder>
```

### Generate a report

```bash
uv run src/pipeline/report.py <evaluation-file-or-folder> --output-dir <folder>
```

## Project structure

```
src/pipeline/
    translation.py       # Sends documents to the Zia translation service
    evaluation.py        # Scores translations against originals (LLM judge)
    report.py            # Aggregates evaluation results into Markdown reports
    run_pipeline.py       # Chains the three stages above
    word_count.py         # Deterministic word counting used by evaluation.py
    schemas.py            # Shared data models
docs/
    command_examples.md   # More CLI examples
```
