# Constructing Reliable Social Networks from Conversational Data

This repository contains the code and processed data accompanying the paper
*Constructing Reliable Social Networks from Conversational Data: An Ensemble
Prompt Engineering Approach with Uncertainty Quantification*.

Raw classroom transcripts are not included because of privacy and
confidentiality restrictions. The retained model predictions are anonymized and
exclude raw utterance text, model reasoning, and API credentials. Every student
identifier has been replaced with a stable pseudonym `s01`-`s22` (teacher
utterances are marked `T`); the pseudonyms are the node labels in the network
matrices and the join key in `Data/scores.csv`.

## Repository structure

```text
Code/Python/            LLM classification and logit extraction
Code/Python/talkmoves/  TalkMoves external-criterion evaluation
Code/R/                 Main and Supplement network analyses
Data/                   Final EXP/EOI networks and student scores
Rpackage/               Source for the custom nbamen package
Results/                Generated tables and figures
```

## Python classification code

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The primary scripts are:

- `Code/Python/main_analysis.py`: reconstructs the retained five-model
  ensemble, reproduces main-text agreement and entropy results, and verifies
  both submitted network matrices.
- `Code/Python/supplement_analysis.py`: reproduces the human-validation
  agreement tables and figures for the Supplement.
- `Code/Python/llm.py`: classification with a local Transformers model.
- `Code/Python/llm_api.py`: classification through OpenAI, Anthropic, or
  Google APIs. Copy `apikey.json.template` to `apikey.json` and fill in the
  `openai_api_key` / `anthropic_api_key` / `google_api_key` field for the
  provider you pass to `--provider` (the `api_key` field is only for the
  optional OpenRouter backend). `apikey.json` is git-ignored.
- `Code/Python/llm_logit.py`: token-probability extraction from an existing
  classification CSV containing model reasoning.

All three scripts provide their current arguments through `--help`. For
example:

```bash
python Code/Python/llm.py --help
python Code/Python/llm_api.py --help
python Code/Python/llm_logit.py --help
```

The retained results can be reproduced without API access:

```bash
python Code/Python/main_analysis.py
python Code/Python/supplement_analysis.py
python Code/Python/talkmoves/talkmoves_analysis.py
```

Users must supply their own DOCX transcript directory when running the
classification scripts. The retained final five-model predictions and ensemble
are provided under `Code/Python/data/`; anonymized human annotations are under
`Code/Python/human+labeling/`. Optional vLLM and OpenRouter implementations are
kept under `Code/Python/optional_backends/` and are not required for the retained
analyses.

## TalkMoves external criterion

`Code/Python/talkmoves/talkmoves_analysis.py` reproduces the *External Criterion
Evaluation on TalkMoves* results (main text and Supplement): five-class and
collapsed three-class performance of the five-model ensemble against the
TalkMoves expert reference labels, and the accuracy gap between unanimous and
split-vote items. It runs from the retained per-model predictions with no API
access and writes to `Results/TalkMoves/`.

```bash
python Code/Python/talkmoves/talkmoves_analysis.py
```

Inputs are under `Code/Python/data/talkmoves/`. TalkMoves \[Suresh et al., 2022]
is redistributed there under CC BY-NC-SA 4.0; see `Code/Python/data/talkmoves/NOTICE`.
Those files are not covered by this repository's `LICENSE`. See
`Code/Python/talkmoves/README.md` for details.

## R network analyses

Run the R scripts from the repository root. Install the required packages with:

```r
install.packages(c(
  "coda", "dplyr", "future", "furrr", "ggplot2", "ggrepel",
  "gridExtra", "htmlwidgets", "igraph", "MCMCpack", "plotly",
  "purrr", "tidyr",
  "Rcpp", "RcppArmadillo"          # build-time dependencies for nbamen
))
install.packages(
  "Rpackage/nbamen_0.2.1.tar.gz",
  repos = NULL,
  type = "source"                  # needs a C++ toolchain
)
```

The scripts use eight parallel workers by default and perform full 100,000-
iteration MCMC fits, so execution can require substantial time and memory.

### Main analysis

```bash
Rscript Code/R/main.R
```

`main.R` reads `Data/Network_EXP.csv`, `Data/Network_EOI.csv`, and
`Data/scores.csv`; computes the centrality tables; fits ten BIC-selected AMEN
chains per network; verifies that `gamma = 1`; and calculates multichain network
mediation. Tables, network figures, the EXP latent-position PNG, and interactive
EOI latent-position HTML views are written under `Results/Main/`.

### Supplement analysis

```bash
Rscript Code/R/supplement.R
```

`supplement.R` independently performs the BIC dimension and dispersion-prior
searches, fits ten selected chains per network, computes convergence diagnostics
and mediation summaries, and evaluates all 18,000 retained posterior-position
draws. Outputs are written under `Results/Supplement/`.

The scripts save final tables and figures, not the large fitted-model RDS
objects. `Code/R/utils.R` contains the shared analysis and plotting functions.

## Processed analysis inputs

- `Data/Network_EXP.csv`: directed weighted Explanation network.
- `Data/Network_EOI.csv`: directed weighted Engage Others' Ideas network.
- `Data/scores.csv`: one row per student. Columns: `csts` (pre-test / prior
  achievement, also the treatment split at 350), `test_total2` (post-test
  outcome), `class` (section id), `s_l_name` (student pseudonym, the join key),
  `gender` (0/1 covariate).

## License

This project is distributed under the terms in `LICENSE`. The exception is
`Code/Python/data/talkmoves/`, which contains files derived from the TalkMoves
dataset and is redistributed under CC BY-NC-SA 4.0 as described in
`Code/Python/data/talkmoves/NOTICE`.
