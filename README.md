# LLM-Symbolic-Handover

`LLM-Symbolic-Handover` is a research-oriented extension of the handover framework from `HandoverOptimDRL`. It keeps the original 3GPP and PPO handover evaluation pipeline, and adds an LLM-assisted symbolic layer for state abstraction, rule extraction, rule deployment, repeated experiments, aggregation, and plotting.

This repository is intended for experiments on explainable reinforcement learning for mobile handover control, especially when the goal is to transform PPO behaviour into auditable symbolic rules.

## What This Repository Adds

Compared with the upstream handover project, this repository adds:

- LLM-assisted symbolic mapper generation from PPO state/action interfaces
- symbolic knowledge-base construction during PPO rollout
- rule extraction as a symbolic decision list
- rule-based deployment on the handover environment
- repeated LLM-run evaluation, result aggregation, and plotting utilities

## Lineage And Acknowledgement

This repository is directly derived from the following upstream codebase:

 `HandoverOptimDRL`
   - Repository: `https://github.com/kit-cel/HandoverOptimDRL`
   - Role in this project: direct code ancestor for the handover environment, PPO evaluation pipeline, data layout, and baseline validation flow. The processed RSRP and SINR datasets under `data/processed/` also originate from the upstream `HandoverOptimDRL` simulation pipeline.

## Repository Layout

```text
.
|-- data/
|   |-- processed/                     # Processed RSRP / SINR MATLAB traces
|-- models/
|   |-- model.zip                      # PPO model artifact used by validation scripts
|-- scripts/
|   |-- validate_3gpp.py              # 3GPP baseline evaluation
|   |-- generate_llm_symbol_mapper.py # Stage 1: build symbolic mapper
|   |-- validate_ppo.py               # PPO evaluation + symbolic rule extraction
|   |-- validate_ppo_llm_rule.py      # Deploy extracted symbolic rules
|   |-- batch_experiment_llm_rule.py  # Multi-run experiment driver
|   |-- aggregate_llm_results.py      # Aggregate repeated LLM-run outputs
|   |-- plot_results.py               # Generate figures from saved NPZ files
|   |-- symbolic_module_llm.py        # Symbolic rule extraction / deployment logic
|-- src/ho_optim_drl/                 # Environments, protocols, config, loader, utils
|-- setup.py
|-- requirements.txt                 # Python package dependencies
|-- README.md
|-- LICENSE
```

## Single-Run And Batch Scripts

This repository is organized into two script groups:

- Core single-run scripts:
  - `scripts/validate_3gpp.py`
  - `scripts/generate_llm_symbol_mapper.py`
  - `scripts/validate_ppo.py`
  - `scripts/validate_ppo_llm_rule.py`

  These scripts expose the main logic of the project and are the best entry point for understanding or debugging the pipeline.

- Batch and aggregation scripts:
  - `scripts/batch_experiment_llm_rule.py`
  - `scripts/aggregate_llm_results.py`
  - `scripts/plot_results.py`

  These scripts are intended for repeated experiments, post-processing, and producing figures suitable for reports or papers.

## Installation

```bash
git clone https://github.com/yyh2005-work/LLM-Symbolic-Handover
cd LLM-Symbolic-Handover
pip install -r requirements.txt
pip install -e .
```

## Data And Model Prerequisites

The evaluation scripts expect the following assets to already exist:

- processed `RSRP` and `SINR` traces under `data/processed/`
- a PPO model artifact under `models/model.zip`

Default evaluation speeds are:

- `30 km/h`
- `50 km/h`
- `70 km/h`
- `90 km/h`

If you need to regenerate the radio traces or revisit the original simulation setup, follow the upstream `HandoverOptimDRL` data-generation workflow.

## Recommended Execution Order

The symbolic pipeline is stage-based. In particular, `validate_ppo.py` expects a pre-generated symbolic mapper to exist under `results/llm_generated_symbxrl/stage1/`.

### 1. Run The 3GPP Baseline

```bash
python scripts/validate_3gpp.py
```

This produces the 3GPP comparison output used later by the plotting script.

### 2. Generate The Symbolic Mapper

LLM mode:

```bash
set DEEPSEEK_API_KEY=<YOUR_API_KEY>
python scripts/generate_llm_symbol_mapper.py
```

### 3. Run PPO Validation And Extract Rules

```bash
python scripts/validate_ppo.py
```

This step evaluates the PPO agent, records symbolic state-action observations, saves a symbolic knowledge base, and exports a deployable decision list.

### 4. Deploy The Rule-Based Agent

```bash
python scripts/validate_ppo_llm_rule.py
```

This step reuses the generated mapper and decision list to execute a rule agent directly on the handover environment.

### 5. Plot Single-Run Comparison Figures

```bash
python scripts/plot_results.py
```

This generates figures under `results/figures/`.

## Repeated Experiments

If you want multiple successful LLM-generated runs before aggregating results:

```bash
python scripts/batch_experiment_llm_rule.py
python scripts/aggregate_llm_results.py
python scripts/plot_results.py
```

The batch runner repeatedly calls the mapper-generation stage until the requested number of successful true-LLM runs is reached. The aggregation script then computes mean metrics across successful runs and prepares data for plotting.