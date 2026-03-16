# Discourse-Driven Evaluation: Unveiling Factual Inconsistency in Long Document Summarization

This repository contains code and resources for the NAACL 2025 paper
["Discourse-Driven Evaluation: Unveiling Factual Inconsistency in Long Document Summarization"](https://aclanthology.org/2025.naacl-long.103/).

## Overview

Detecting factual inconsistency in long-document summarization is challenging because both source documents and summaries contain complex discourse structure. This project studies how discourse phenomena relate to factual inconsistency and introduces a discourse-driven evaluation framework that:

- decomposes long texts into discourse-inspired chunks,
- scores chunk-level or sentence-level consistency with NLI-based models,
- aggregates these signals with discourse-aware structure,
- improves factuality evaluation on long-document summarization benchmarks.

This repository is being organized into a clean research codebase. The initial scaffold focuses on two main directories:

- `data/`: datasets, metadata, and preprocessing artifacts used in experiments.
- `structscore/`: the implementation of the proposed approach and related utilities.
- `DMRST/`: the bundled discourse parser used to obtain EDU segmentation and RST tree outputs.

## Repository Structure

```text
.
├── DMRST/
│   ├── README.md
│   ├── requirements.txt
│   └── sample_infer.py
├── data/
│   └── README.md
├── structscore/
│   ├── __init__.py
│   ├── README.md
│   └── notebooks/
│       └── README.md
├── LICENSE
└── README.md
```

## Current Status

This is a starter version of the repository structure. The code is being migrated from working notebooks into a cleaner layout.

Planned additions:

- notebook cells and scripts for the proposed method,
- dataset preparation instructions,
- model inference and evaluation scripts,
- reproducibility details for experiments. DIVERSUMM [DONE]

## Parsing Pipeline with DMRST

This project leverages the bundled [`DMRST`](./DMRST) parser to parse documents or summaries into discourse units and RST tree structures before applying the proposed evaluation framework.

The expected workflow is:

1. Set up `DMRST` first by following the instructions in [`DMRST/README.md`](./DMRST/README.md).
2. Install the parser dependencies from [`DMRST/requirements.txt`](./DMRST/requirements.txt).
3. Prepare your own input data.
4. Run the DMRST inference script to extract discourse segmentation and parsing outputs.
5. Use those RST outputs in the downstream `structscore/` pipeline.

## Using Your Own Data

The helper script [`DMRST/sample_infer.py`](./DMRST/sample_infer.py) shows how we run DMRST on a CSV file of examples.

It expects a CSV file with at least these columns:

- `source`
- `summary`

The script currently parses the `summary` field and writes a new file with parser outputs added. In particular, it appends:

- `summ_sents`
- `summ_segments`
- `summ_tree_parsing`

and saves the result to a new file ending in `_parsed.csv`.

Example usage:

```bash
cd DMRST
python sample_infer.py --data_path /path/to/your_data.csv
```

The parser uses the DMRST checkpoint and `xlm-roberta-base` backbone defined in the bundled parser code. Since DMRST is a separate parser component, the first step for new users should always be to get DMRST working correctly by following the official DMRST setup and inference instructions before processing their own datasets in this repository.

## Data

Use the `data/` directory for any datasets or derived artifacts used in the paper, such as:

- raw benchmark files,
- processed inputs for evaluation,
- split definitions,
- intermediate outputs and cached annotations.

Please avoid committing large proprietary or restricted datasets directly if redistribution is not allowed. Instead, include download instructions and preprocessing scripts when needed.

## StructScore

The `structscore/` directory is intended to hold the implementation of the proposed discourse-driven factuality evaluation framework. As the notebook code is organized, this folder can be expanded with modules for:

- [] discourse-based chunking,
- score aggregation from DMRST-derived discourse outputs,
- experiment runners and evaluation utilities.

## Citation

If you use this repository or build on this work, please cite:

```bibtex
@inproceedings{zhong-litman-2025-discourse,
  title = "Discourse-Driven Evaluation: Unveiling Factual Inconsistency in Long Document Summarization",
  author = "Zhong, Yang and Litman, Diane",
  booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  year = "2025",
  pages = "2050--2073",
  address = "Albuquerque, New Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.naacl-long.103/",
  doi = "10.18653/v1/2025.naacl-long.103"
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
