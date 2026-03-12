# Dataset Collection Agent

Script-driven SGR agent for multi-source classification dataset collection (image and text).

## Features

- Uses `SGRToolCallingAgent` with custom tools.
- Searches image and text datasets on Hugging Face and Kaggle.
- Supports modality selection via `--modalities image,text`.
- Selects datasets directly from discovered candidates (no verification/filtering step).
- Downloads each accepted dataset to its own directory under `data/`.
- If class coverage is insufficient, runs Tavily fallback with one query per missing class and downloads web images.
- Writes a JSON manifest with selected/rejected datasets and fallback provenance.

## Environment Variables

- `OPENAI_API_KEY` (or `SGR__LLM__API_KEY`) - required
- `TAVILY_API_KEY` (or `SGR__SEARCH__TAVILY_API_KEY`) - optional, enables Tavily fallback
- `HF_TOKEN` - optional, for Hugging Face access
- Kaggle credentials (`~/.kaggle/kaggle.json`) - required for Kaggle API

## Run

```bash
python main.py \
  --query "collect a dataset for 3 swan classes: mute swan, trumpeter swan, whooper swan" \
  --data-dir data \
  --manifest-path reports/collection_manifest.json
```

With explicit class override:

```bash
python main.py \
  --query "find swan datasets" \
  --classes "mute swan, trumpeter swan, whooper swan"
```

## CLI Options

- `--query` (required)
- `--classes`
- `--modalities` (default: `image,text`)
- `--config` (default: `config.yaml`)
- `--agent` (default: `dataset_collection_agent`)
- `--data-dir` (default: `data`)
- `--manifest-path` (default: `reports/collection_manifest.json`)
- `--max-datasets` (default: `20`)
- `--max-web-images-per-class` (default: `300`)
- `--tavily-max-results` (default: `5`)
- `--log-level` (`DEBUG|INFO|WARNING|ERROR`, default: `INFO`)
- `--log-file` (default: `logs/run-<timestamp>.log`)

## Config

`config.yaml` contains:
- SGR model/search/execution settings
- tool registry definitions
- agent definition and prompt file paths

## Notes

- No tests are created or modified.
- Verification filtering is disabled; datasets are selected directly from discovered candidates.
- Manifest generation is enforced by the SGR tool workflow.
- Runtime logs are streamed to console and written to a file for each run.
