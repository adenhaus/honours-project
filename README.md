
# Improving Faithfulness of Multilingual Table-to-Text NLG in African Languages

## Overview

This is the repository for my undergraduate <a href="http://example.com/](http://www.drps.ed.ac.uk/20-21/dpt/cxinfr10044.htm" target="_blank">honours project</a> [honours project](http://www.drps.ed.ac.uk/20-21/dpt/cxinfr10044.htm) (AKA dissertation) which I completed in my fourth and final year of my BSc Computer Science at [The University of Edinburgh](https://www.ed.ac.uk/informatics). This research project, titled **"Improving Faithfulness of Multilingual Table-to-Text Natural Language Generation in African Languages."** aims to improve the faithfulness of multilingual Table-to-Text natural language generation by introducing intermediate text plans, or blueprints, comprised of question-answer pairs.

## Research inspiration

This project draws mainly from these two papers:
- [TaTA: A Multilingual Table-to-Text Dataset for  African Languages (Gehrmann et al., 2023)](https://aclanthology.org/2023.findings-emnlp.118/)
- [Conditional Generation with a Question-Answering Blueprint (Narayan et al., 2023)](https://aclanthology.org/2023.tacl-1.55/)

## Results summary

Todo.

## Project Structure

The repository is organised as follows:

- **`code/`**: This directory contains all the code related to the project. It includes the Jupyter notebooks I used to process the data and run evaluations (run in Google Colab), as well as Python scripts and Bash files, mostly for training Transformer models (run on my Edinburgh University's School of Informatics' [research cluster](https://computing.help.inf.ed.ac.uk/research-cluster)).

- **`data/`**: Here, you can find the datasets used for experimentation. The dataset files are in CSV format, and each dataset is placed in a separate folder for clarity. These are derived from Google Research's [TaTA](https://github.com/google-research/url-nlp/tree/main/tata).

- **`hons-project.pdf`**: My write-up and submission.

## Using the models

The best-performing fine-tuned model is publicly available on my [🤗 Hugging Face profile](https://huggingface.co/adenhaus). You can use it with the [🤗 Transformers](https://huggingface.co/docs/hub/transformers) library:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("adenhaus/mt5-large-stata")
model = AutoModelForSeq2SeqLM.from_pretrained("adenhaus/mt5-large-stata")
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
