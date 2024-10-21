
# Multilingual Table-to-Text Generation with Question-Answer Plans

## Overview

This is the repository for my undergraduate [honours project](http://www.drps.ed.ac.uk/20-21/dpt/cxinfr10044.htm) (AKA dissertation) which I completed in the fourth and final year of my BSc Computer Science at [The University of Edinburgh](https://www.ed.ac.uk/informatics). This research project, titled **"Multilingual Table-to-Text Generation with Question-Answer Plans"** aims to improve the faithfulness of multilingual Table-to-Text Natural Language Generation by introducing intermediate text plans, or blueprints, comprised of question-answer pairs.

## Research inspiration

This project builds mainly upon these two papers:
- [TaTA: A Multilingual Table-to-Text Dataset for  African Languages (Gehrmann et al., 2023)](https://aclanthology.org/2023.findings-emnlp.118/)
- [Conditional Generation with a Question-Answering Blueprint (Narayan et al., 2023)](https://aclanthology.org/2023.tacl-1.55/)

## Results summary

The results show QA plans improve the factuality of Table-to-Text outputs in English, but do not result in similar gains in the multilingual setting. This is due to A) innacuriacies in machine translating the QA plans, generated in English, into the target languages, and B) the models struggling to rely heavily on the blueprints they generate. A detailed analysis of this is conducted in the project.

## Project Structure

The repository is organised as follows:

- **`hons-project.pdf`**: My write-up and submission.

- - **`hons-project-short.pdf`**: A shortened version of the report in ACL style.

- **`training_scripts/`**: This directory contains Python scripts and Bash files, mostly for training Transformer models (run on Edinburgh University's School of Informatics' [research cluster](https://computing.help.inf.ed.ac.uk/research-cluster)).

- **`datasets/`**: Here, you can find the datasets used for experimentation. The dataset files are in CSV format, and each dataset is placed in a separate folder for clarity. These are derived from Google Research's [TaTA](https://github.com/google-research/url-nlp/tree/main/tata).

- **`data_processing_notebooks/`**: This directory contains the Jupyter notebooks I used to process the data and run evaluations (run in Google Colab).
  - Evaluation:
    - `prediction_suite`: Uses finetuned models to generate a predictions file from the test set.
    - `eval_suite`: Uses automatic metrics to evaluate model predictions.
  - Blueprint generator pipeline:
    - `QA-generator`: Generates and heuristically filters QA pairs to create a blueprint for all English samples.
    - `BlueprintFiller`: Extends blueprints generated on English samples to all other parallel samples (takes output of `QA-generator` as input).
    - `TranslateBlueprints`: Translates blueprints into target langauges (takes output of `BlueprintFiller` as input).
    - `AddFullBlueprints`: Combines blueprint and verbalisation into new target (takes output of `TranslateBlueprints` or `QA-generator` as input).
  - Metric Pearson correlations:
    - `BertScore`: Calculates the Pearson correlation of BERTScore with human evaluations on TaTA.
    - `FactKB`: Calculates the Pearson correlation of FactKB with human evaluations on TaTA.
    - `STATA`: Calculates the Pearson Correlation of the StATA on the test set.
    - `p_value`: Calculates the Pearson correlation and p-value from a txt file os scores.
  - Other preprocessing:
    - `ExplodeRows`: Explodes rows in TaTA along each row's multiple references to increase the dataset size.
    - `ExtractSplitsStata`: Randomly sample train/dev/test splits from the human annotations file.
    - `json_to_csv`: Convert json files to csv files.
    - `csv-length-checker`: Checks the length of csv files.
    - `SplitTestLangs`: Split the test set into individual languages.
    - `MakeEnglishDataset`: Removes non-English examples from the dataset to create an English subset.
    - `SaveModel`: Download model from HF and save it locally.
  - Dataset stats:
    - `BlueprintSimilarity`: Calculates BLEU and chrF similarity between tables, blueprints and verbalisations.
    - `BlueprintStats`: Calculates statistics about QA blueprints like the average number of QA pairs in a blueprint.
    - `google-trans`: Calculate how well Google Translate performs on the dataset.
    - `LengthStats`: Calculates lengths of topkenised inputs and references in the datasets.
    - `PlotStataLoss`: Plot loss curves.

## Using the models

The best-performing fine-tuned model is publicly available on my [🤗 Hugging Face profile](https://huggingface.co/adenhaus). You can use it with the [🤗 Transformers](https://huggingface.co/docs/hub/transformers) library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("adenhaus/mt5-large-tata")
model = AutoModelForSeq2SeqLM.from_pretrained("adenhaus/mt5-large-tata")
```

## Running the code

Follow these steps to get started with running the code for project:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/adenhaus/hons-project.git
    cd hons-project
    ```

2. **Install Dependencies:**
    Ensure you have the necessary dependencies installed. You can find the required libraries in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3. **Notebooks:**
    I ran these in Google Colab. Some will benefit from the use of Colab's T4 GPUs.

4. **Training scripts:**
    The Bash scripts are written to work on a [slurm](https://computing.help.inf.ed.ac.uk/slurm) compute cluster, and all are run with a single A40 GPU.

## Acknowledgments

I would like to express my gratitude to my supervisor, [Mirella Lapata](https://scholar.google.co.uk/citations?user=j67B9Q4AAAAJ&hl=en) for her guidance and support throughout the course of this project; [Sebastian Ruder](https://scholar.google.com/citations?user=8ONXPV8AAAAJ&hl=en), one of the authors of the TaTA dataset, who gave me some valuable pointers while we were both at Google; and [Tom Sherborne](https://scholar.google.com/citations?user=50nZ2yAAAAAJ&hl=en), who helped me a great deal with technical questions.
