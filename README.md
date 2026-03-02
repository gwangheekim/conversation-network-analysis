# Constructing Reliable Social Networks from Conversational Data: An Ensemble Prompt Engineering Approach with Uncertainty Quantification

This repository contains the code and data accompanying the paper *"Constructing Reliable Social Networks from Conversational Data: An Ensemble Prompt Engineering Approach with Uncertainty Quantification"*.

The project combines Python code for LLM-based dialogue classification with R code for network analysis using AMEN models and network mediation.

## Note on Data Availability

Due to privacy and confidentiality constraints, the **raw transcript (DOCX) files** used in the original analysis are **not included** in this submission. The Python classification scripts (`llm.py`, `llm_api.py`) reference transcript directories as command-line arguments; users must supply their own data when running these scripts.

The intermediate classification results are provided in `Code/Python/data/` with the raw utterance text removed to protect participant privacy. Similarly, human labeling data in `Code/Python/human+labeling/` has been anonymized — labeler identities are replaced with anonymous IDs (P1–P8) and utterance text has been removed, retaining only the classification labels and metadata necessary for validation.

> **Note**: The `--transcript_dir` argument defaults to `./transcripts/`. Always specify your own transcript directory path when running these scripts.

## Directory Structure

```
├── Code/
│   ├── Python/                 # LLM-based dialogue classification
│   │   ├── data/               # Intermediate LLM classification results (utterance text removed)
│   │   └── human+labeling/     # Human labeler annotations (anonymized, utterance text removed)
│   └── R/                      # Network analysis and mediation models
├── Data/                       # Processed adjacency matrices and student scores
├── Rpackage/                   # Custom R package for AMEN model
└── README.md                   # This file
```

## 1. Python Code (`Code/Python/`)

The Python code handles dialogue data loading and classification using LLMs.

### Core Files:

#### `llm.py`
- **Purpose**: Dialogue classification using open-source LLM models
- **Contents**: 
  - Extracts dialogue from DOCX transcript files
  - Applies prompt-based classification using transformers
  - Supports various open-source models (e.g., Llama, Qwen)
- **Usage**:
  ```bash
  python llm.py --model_id "[model name]" --output_csv "[output file name]" --transcript_dir "[data path]"
  ```

#### `llm_api.py`
- **Purpose**: Dialogue classification using commercial LLM APIs
- **Contents**:
  - Same core Contents as `llm.py` but uses commercial APIs
  - Supports OpenAI, Anthropic, and Google APIs
  - Handles API rate limiting and error management
- **Usage**:
  ```bash
  python llm_api.py --provider "[provider (google, openai, anthropic)]" --apikey_json "[apikey path]" --model_name "[model name]" --transcript_dir "[data path]" --output_csv "[output file name]" --sleep_sec [seconds]
  ```
  - `--sleep_sec`: Optional seconds to wait between API calls for rate-limit throttling (default: 0.0)

#### `llm_logit.py`
- **Purpose**: Extract model logits for uncertainty analysis
- **Contents**:
  - Computes logit values from open-source models
  - Used for uncertainty quantification and model confidence analysis
  - Provides detailed probability distributions for classifications
- **Arguments**: `--input_csv` (path to a previous classification result CSV), `--model_id`, `--output_csv`
- **Usage**:
  ```bash
  python llm_logit.py --input_csv "classifications.csv" --model_id "[model name]" --output_csv "classifications_with_logits.csv"
  ```

### Validation and Preprocessing Notebooks:

#### `validation.ipynb`
- **Purpose**: Comprehensive validation and uncertainty analysis
- **Contents**:
  - Model performance metrics and accuracy evaluation
  - Uncertainty analysis using model logits (when available)
  - Comparison with human labeler annotations
  - Cross-validation results between different LLM approaches
  - Statistical validation of classification reliability
- **Usage**: Open and run in Jupyter Notebook or JupyterLab

#### `preprocess.ipynb`
- **Purpose**: Data transformation for network analysis
- **Contents**:
  - Processes LLM classification outputs
  - Prepares data format compatible with R network analysis
  - Handles data cleaning and validation
- **Usage**: Open and run in Jupyter Notebook or JupyterLab

## 2. R Code (`Code/R/`)

The R code implements network analysis using AMEN models and network mediation.

> **Note**: R scripts assume the **repository root** as the working directory, e.g., `setwd("path/to/this/repo")`.

### R Dependencies

```r
install.packages(c("ggplot2", "gridExtra", "dplyr", "plotly", "igraph",
                   "ggrepel", "RColorBrewer", "knitr", "purrr", "tidyverse",
                   "MCMCpack", "coda", "future", "furrr", "progressr",
                   "future.apply", "tidyr"))
# Install the custom nbamen package
install.packages("./Rpackage/nbamen_0.1.0.tar.gz", repos = NULL, type="source")
```

### Core Files:

#### `main.R`
- **Purpose**: Main analysis pipeline
- **Contents**:
  - Loads and preprocesses adjacency matrix data
  - Fits AMEN models for both Explanation (EXP) and Elaboration of Ideas (EOI) networks
  - Performs network mediation analysis
  - Conducts sensitivity analysis

#### Key Analysis Steps:
1. **AMEN Model Fitting**: 
   - Fits negative binomial AMEN models
   - Uses multiple chains for robustness (10 runs each)
   - Applies Procrustes matching for chain alignment
2. **Network Mediation**: Computes Natural Direct Effects (NDE) and Natural Indirect Effects (NIE)
3. **Sensitivity Analysis**: Posterior sampling for uncertainty quantification

#### `utils.R`
- Utility functions for data processing, model post-processing, and visualization

## 3. R Package (`Rpackage/`)

### `nbamen` Package
- **Purpose**: Custom R package implementing AMEN models for count data with negative binomial distribution
- **Key Features**:
  - MCMC algorithms for network model estimation
  - Comprehensive model diagnostics (BIC, WAIC, DIC)
  - Procrustes matching for post-processing

#### Installation:
```r
install.packages("./Rpackage/nbamen_0.1.0.tar.gz", repos = NULL, type="source")
library(nbamen)
```

#### Main Function:
- `amen_count_nb()`: Fits AMEN model with negative binomial distribution

## 4. Data (`Data/`)

### Current Files:
- **`Network_EOI.csv`**: Adjacency matrix for Engage other's idea network
- **`Network_EXP.csv`**: Adjacency matrix for Explanation network  
- **`scores.csv`**: Student math scores

## 5. Intermediate Data (`Code/Python/data/` and `Code/Python/human+labeling/`)

These directories contain intermediate results from the LLM classification pipeline. Raw utterance text has been removed from all files to protect participant privacy.

### `Code/Python/data/`
Contains LLM classification output CSV files generated by `llm.py`, `llm_api.py`, and `llm_logit.py`. Each file includes classification labels, engagement levels, and reference information for each utterance. Logit files additionally contain token-level probability distributions used for uncertainty quantification.

### `Code/Python/human+labeling/`
Contains human labeler annotation data organized by anonymized labeler ID (`labeler_P1` through `labeler_P8`). Each labeler's directory includes CSV files with classification labels (matching the LLM classification categories) for the same set of dialogue sessions, used for inter-rater reliability analysis in `validation.ipynb`.

## Execution Workflow

### Complete Analysis Pipeline:

1. **Dialogue Classification** (Python):
   ```bash
   # Using open-source model
   python Code/Python/llm.py --model_id "meta-llama/Llama-3.1-8B-Instruct" --output_csv "classifications.csv" --transcript_dir "path/to/transcripts"
   
   # Or using API
   python Code/Python/llm_api.py --provider "openai" --apikey_json "keys.json" --model_name "gpt-4.1" --transcript_dir "path/to/transcripts" --output_csv "classifications.csv"
   ```

2. **Logit Extraction** (Python, optional for open-source models):
   ```bash
   # Extract logits for uncertainty quantification 
   python Code/Python/llm_logit.py --input_csv "classifications.csv" --model_id "meta-llama/Llama-3.1-8B-Instruct" --output_csv "classifications_with_logits.csv"
   ```

3. **Validation** (Python):
   - Open and run `Code/Python/validation.ipynb` in Jupyter to:
     - Evaluate LLM classification consistency
     - Analyze uncertainty using logits (if available)
     - Compare results with human labeler annotations

4. **Data Preprocessing** (Python):
   - Open and run `Code/Python/preprocess.ipynb` in Jupyter to convert validated classifications to adjacency matrices

5. **Network Analysis** (R):
   ```r
   # Install package
   install.packages("./Rpackage/nbamen_0.1.0.tar.gz", repos = NULL, type="source")
   
   # Execute main analysis code
   # Run the code in Code/R/main.R step by step
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
