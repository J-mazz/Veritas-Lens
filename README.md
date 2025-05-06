# Veritas-Lens
Machine Learning model for classifying political bias.
# VeritasLens: Political Bias Detector

## Project Goal

This project aims to train a machine learning model capable of classifying the political bias (Left, Center, Right) of news articles based on their text content. The goal is to leverage modern Natural Language Processing (NLP) techniques, specifically transformer models like BERT, to understand and categorize media bias. This tool could potentially be developed into a user-facing application or browser plugin.

## Datasets and Acknowledgements

Training a robust bias detector requires diverse data. This project utilized a combination of the following datasets sourced from Hugging Face:

1.  **`siddharthmb/article-bias-prediction-media-splits`**:
    * Source: Data originally from Ad Fontes Media.
    * Description: Contains news article content labeled with bias ratings.
    * Citation:
        ```
        @inproceedings{maheshwari-etal-2023-detecting,
            title = "Detecting Political Bias in News Articles using Natural Language Processing",
            author = "Maheshwari, Siddharth and
              A, Abhilash",
            editor = "Gurevych, Iryna and
              Hovy, Eduard and
              Huang, Xuanjing",
            booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
            month = may,
            year = "2023",
            address = "Dubrovnik, Croatia",
            publisher = "Association for Computational Linguistics",
            url = "[https://aclanthology.org/2023.findings-eacl.48](https://aclanthology.org/2023.findings-eacl.48)",
            pages = "648--654",
        }
        ```
    * HF Link: [https://huggingface.co/datasets/siddharthmb/article-bias-prediction-media-splits](https://huggingface.co/datasets/siddharthmb/article-bias-prediction-media-splits)

2.  **`Faith1712/Allsides_political_bias_proper`**:
    * Source: Data scraped from AllSides.com by the dataset creator.
    * Description: Contains news article content with bias ratings ('Left', 'Center', 'Right') according to AllSides' methodology.
    * Attribution: Data originates from AllSides.com. Dataset prepared by Hugging Face user Faith1712.
    * HF Link: [https://huggingface.co/datasets/Faith1712/Allsides_political_bias_proper](https://huggingface.co/datasets/Faith1712/Allsides_political_bias_proper)

3.  **`cajcodes/political-bias`**:
    * Source: Provided by Hugging Face user cajcodes.
    * Description: Contains news text labeled with bias ('left', 'center', 'right').
    * Attribution: Dataset prepared by Hugging Face user cajcodes.
    * HF Link: [https://huggingface.co/datasets/cajcodes/political-bias](https://huggingface.co/datasets/cajcodes/political-bias)

*Note: Only articles labeled explicitly as 'Left', 'Center', or 'Right' (case-insensitive) were used from these datasets for the final 3-class classification task.*

## Preprocessing Pipeline

A multi-stage preprocessing pipeline was developed:

1.  **Individual Dataset Processing (`preprocess_hf_generic.py`):**
    * Loaded each Hugging Face dataset.
    * Identified and extracted relevant text and label columns (handling inconsistencies like `'label'` vs. `'bias_rating'`).
    * Handled different label formats (strings, ClassLabel integers) and mapped them to the standardized labels: 'left', 'center', 'right'.
    * Applied basic text cleaning (lowercase, remove URLs/emails, keep basic punctuation).
    * Saved each processed dataset as an intermediate CSV file.
2.  **Combination and Splitting (`preprocess_combine_datasets.py`):**
    * Loaded the intermediate CSV files.
    * Concatenated them into a single large DataFrame.
    * Filtered out any remaining non-standard labels or empty text.
    * Shuffled the combined data.
    * Performed a stratified train-validation-test split (80%/10%/10%).
    * Saved the final splits into a directory structure (`train/left/*.txt`, `train/center/*.txt`, etc.) suitable for TensorFlow's `text_dataset_from_directory`.

## Model Architecture and Training

* **Model:** `bert-base-uncased` fine-tuned for sequence classification using the `TFBertForSequenceClassification` model from the Hugging Face `transformers` library.
* **Fine-Tuning:** The entire pre-trained BERT model was unfrozen and fine-tuned along with the classification head.
* **Training Framework:** TensorFlow with Keras API.
* **Training Loop:** A custom training loop was implemented (`train_model_final_combined.py`) to provide more control and overcome framework-specific issues encountered during development.
* **Data Loading:** `tf.data.Dataset` API with `text_dataset_from_directory`, `.cache()`, and `.prefetch()` for efficiency. Tokenization was performed using `tf.py_function` wrapping the Hugging Face tokenizer.
* **Key Hyperparameters:**
    * Optimizer: AdamW (Weight Decay = 0.01)
    * Learning Rate: 2e-5
    * Batch Size: 64 (Optimized for A100 GPU)
    * Max Sequence Length: 256
    * Epochs: 10 (with early stopping patience = 3, stopped after epoch 10)
    * Precision: Single Precision (Float32)

## Development Challenges & Solutions

This project involved significant debugging and iterative refinement:

* **Initial Training Stagnation:** Training only the classification head on a frozen base model failed to learn effectively, indicating the pre-trained features alone were insufficient for this nuanced task.
* **Keras Trainability Issues:** Attempts to fine-tune using Keras subclassing or the Functional API encountered persistent issues where the base model's weights were not recognized as trainable after `model.compile()`. This was diagnosed by checking `len(model.trainable_weights)` and observing misleading model summaries.
* **Solution (Trainability):** Switched to loading the standard `TFBertForSequenceClassification` model directly from Hugging Face, which correctly handles its own trainability settings.
* **Mixed Precision Compatibility:** Attempts to use mixed precision (FP16) with the `LossScaleOptimizer` resulted in `AttributeError: 'Variable' object has no attribute '_distribute_strategy'` during `model.compile()` when used with `TFBertForSequenceClassification`.
* **Solution (Mixed Precision):** Reverted to single precision (Float32) training using the standard AdamW optimizer to ensure stability and successful compilation/training.
* **Data Loading Speed:** Initial file counting using `glob` was slow on Google Drive.
* **Solution (Data Loading):** Switched to using `tf.data.experimental.cardinality` on the loaded dataset to determine the number of batches efficiently.
* **Overfitting:** Initial fine-tuning runs showed signs of overfitting (high training accuracy, poor validation accuracy).
* **Solution (Overfitting):** Combining multiple datasets and switching from standard Adam to the AdamW optimizer (with weight decay) significantly improved generalization.
* **Dataset Column Inconsistencies:** Discovered that different source datasets used different names for text and label columns (`text`/`label`, `content`/`bias_text`, `news_content`/`bias_rating`).
* **Solution (Dataset Columns):** Created a generic preprocessing script (`preprocess_hf_generic.py`) configurable for different column names and standardized the output to 'text' and 'label' before combining.

## Results

* **Training:** The model was trained for 10 epochs on the combined dataset using an NVIDIA A100 GPU.
* **Best Validation Accuracy:** **82.15%** (Achieved at Epoch 8).
* **Test Set Performance:** *(Evaluation pending availability of compute resources)*

    * Test Loss: TBD
    * Test Accuracy: TBD
    * Classification Report: TBD
    * Confusion Matrix: TBD

## Future Work & Deployment Considerations

* Run final evaluation on the test set to get definitive performance metrics.
* Explore conversion to TensorFlow Lite (TFLite) for optimized on-device inference, potentially enabling deployment as a native app or plugin component.
* Investigate potential accuracy improvements (e.g., learning rate scheduling, different base models like RoBERTa).
* Analyze misclassified examples to understand model weaknesses.

## How to Run (Example)

1.  **Setup Environment:**
    ```bash
    pip install tensorflow transformers datasets pandas scikit-learn
    ```
2.  **Run Preprocessing:**
    * Configure and run `preprocess_hf_generic.py` for each source dataset (`siddharthmb/...`, `Faith1712/...`, `cajcodes/...`).
    * Configure and run `preprocess_combine_datasets.py`.
3.  **Run Training:**
    * Ensure `PROCESSED_DATA_DIR` in `train_model_final_combined.py` points to the combined data directory (`.../data/processed_combined`).
    * Run `python train_model.py`. *(Script ID: `train_model`)*
4.  **Run Evaluation:**
    * Ensure saved weights and tokenizer exist from the training run.
    * Ensure `PROCESSED_DATA_DIR` in `evaluate_final_model.py` points to the combined data directory.
    * Run `python evaluate_final_model.py`. *(Script ID: `evaluate_final_model`)*
