# Raw Data Folder

This directory is the designated landing spot for the source CoNLL-U files from the **Zoroastrian Middle Persian Corpus and Dictionary (MPCD)**.

## ⚠️ Licensing and Redistribution Note
To comply with the licensing terms of the MPCD project, the **raw manuscript files are not redistributed in this repository**. 

Users who wish to re-run the feature extraction pipeline (`src/preprocess.py`) must obtain the raw data directly from the official source.

## Instructions for Reproduction

To re-extract the head-modifier pairs:

1.  **Request Access:** Visit [MPCD (Zoroastrian Middle Persian Corpus and Dictionary)](https://mpcorpus.org) to obtain the corpus in **CoNLL-U format**.
2.  **Placement:** Place the `.conllu` files (e.g., `GBd-TD1-01_mptf.conllu`, `DMX-K43a-01_mptf.conllu`, etc.) into this `data/raw/` folder.
3.  **Run Extraction:** Execute the preprocessing script:
    ```bash
    python src/preprocess.py
    ```
    This will generate a new `head_modifier_pairs.csv` in the `data/preprocessed/` directory.

## Expected Format
The extraction script expects standard **Universal Dependencies (UD) CoNLL-U v2** format, including the `Transliteration` and `Transcription` fields in the `MISC` column as provided by the MPCD team.

## Citation for Source Data
If you use the raw corpus files, please cite the MPCD project:
> *The Zoroastrian Middle Persian Corpus and Dictionary (MPCD). Edited by Alberto Cantera, Maria Macuch, and Götz König. https://mpcorpus.org*