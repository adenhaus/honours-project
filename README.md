
# Improving Faithfulness of Multilingual Table-to-Text NLG in African Languages

## Overview

This is the repository for my undergraduate [honours project](http://www.drps.ed.ac.uk/20-21/dpt/cxinfr10044.htm) (AKA dissertation) which I completed in the fourth and final year of my BSc Computer Science at [The University of Edinburgh](https://www.ed.ac.uk/informatics). This research project, titled **"Multilingual Table-to-Text Generation with Question-Answer Plans"** aims to improve the faithfulness of multilingual Table-to-Text Natural Language Generation by introducing intermediate text plans, or blueprints, comprised of question-answer pairs.

## Research inspiration

This project builds mainly upon these two papers:
- [TaTA: A Multilingual Table-to-Text Dataset for  African Languages (Gehrmann et al., 2023)](https://aclanthology.org/2023.findings-emnlp.118/)
- [Conditional Generation with a Question-Answering Blueprint (Narayan et al., 2023)](https://aclanthology.org/2023.tacl-1.55/)

## Results summary

The evidence suggests QA plans improve multilingual Table-to-Text outputs' factuality, but do not result in similar gains in the multilingual setting. This is due to innacuriacies in machine translating the QA plans, generated in English, into the target languages, and the models struggling to rely heavily on the blueprints they generate. A detailed analysis of this is conducted in the project.

## Project Structure

The repository is organised as follows:

- **`training_scripts/`**: This directory contains Python scripts and Bash files, mostly for training Transformer models (run on Edinburgh University's School of Informatics' [research cluster](https://computing.help.inf.ed.ac.uk/research-cluster)).

- **`datasets/`**: Here, you can find the datasets used for experimentation. The dataset files are in CSV format, and each dataset is placed in a separate folder for clarity. These are derived from Google Research's [TaTA](https://github.com/google-research/url-nlp/tree/main/tata).

- **`hons-project.pdf`**: My write-up and submission.

- **`processing_notebooks/`**: This directory contains the Jupyter notebooks I used to process the data and run evaluations (run in Google Colab).

  - `prediction_suite`: Uses finetuned models to generate a predictions file from the test set.
  - `eval_suite`: Uses automatic metrics to evaluate model predictions.
  - `BertScore`: Calculates the Pearson correlation of BERTScore with human evaluations on TaTA.
  - `AddFullBlueprints`:
  - `BlueprintFiller`:
  - `BlueprintSimilarity`:
  - `BlueprintStats`:
  - `csv-length-checker`:
  - `ExplodeRows`:
  - `ExtractSplitsStata`:
  - `FactKB`: Calculates the Pearson correlation of FactKB with human evaluations on TaTA.
  - `google-trans`:
  - `json_to_csv`:
  - `LengthStats`:
  - `MakeEnglishDataset`:
  - `p_value`:
  - `PearsonCorrelation`:
  - `PlotStataLoss`:
  - `QA-generator`:
  - `SaveModel`:
  - `SplitTestLangs`:
  - `STATA`:
  - `TranslateBlueprints`:
  - ``:

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

I would like to express my gratitude to my supervisor, [Mirella Lapata](https://scholar.google.co.uk/citations?user=j67B9Q4AAAAJ&hl=en) for her guidance and support throughout the course of this project.
