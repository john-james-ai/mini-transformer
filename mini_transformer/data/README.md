# Data Package (`mini_transformer/data`)

This package contains all modules responsible for the data lifecycle in the `mini-transformer` project. This includes defining the core data structures, managing data persistence, processing raw data through a pipeline, and handling tokenization.

---

## Core Components

The package is organized around a few key responsibilities, each handled by a specific module or sub-package.

### 1. `dataset.py`

* **Purpose**: Defines the core data structure, `TranslationDataset`.
* **Description**: This dataclass acts as a container for the data (`List[Dict]`), configuration (`DatasetBuilderConfig`), and metrics (`MetricsCollector`). It is an immutable object that represents a specific version of the dataset at a particular stage (e.g., "raw" or "filtered"). It provides a consistent interface for accessing data and its associated metadata.

### 2. `repo.py`

* **Purpose**: Provides a `DatasetRepo` for persisting and retrieving datasets.
* **Description**: The repository abstracts away the storage details. It uses a two-tiered storage strategy:
    * **Object Storage (`ObjectAccessLayer`)**: Stores a "dematerialized" version of the `TranslationDataset` object (metadata only, with an empty `data` list). This serves as a lightweight manifest.
    * **File Storage (`FileAccessLayer`)**: Stores the actual data payload (the list of examples) as a `.jsonl` file.
* This design ensures that listing and querying datasets is fast (as only small manifest objects are read), while the larger data files are only loaded when a specific dataset is explicitly requested via `repo.get()`.

### 3. `builder/`

* **Purpose**: A sub-package containing the logic for processing data in stages.
* **Description**: It follows a Builder design pattern to construct `TranslationDataset` objects.
    * `extractor.py`: Contains `TranslationDatasetBuilderRaw`, which is responsible for the initial extraction of data from a source (like Hugging Face). It downloads the data and computes initial quantitative metrics (e.g., sequence lengths).
    * `data_filter.py`: Contains `TranslationDatasetBuilderFiltered`, which takes a raw `TranslationDataset` and applies filtering rules (e.g., based on length, language ratio) to produce a smaller, cleaner dataset suitable for training.

### 4. `tokenize/`

* **Purpose**: Handles the training and application of text tokenizers.
* **Description**: This sub-package contains the logic for converting text into a numerical format that the model can process.
    * `bpe.py`: Implements `BPETokenization`, a wrapper around the Hugging Face `tokenizers` library to train a Byte-Pair Encoding (BPE) tokenizer on a given corpus. It handles training, saving, and loading the tokenizer artifact.

---

## Workflow

The components in this package are designed to work together in a sequential pipeline:

1.  **Extraction**: A user runs a script that uses `TranslationDatasetBuilderRaw` (`extractor.py`) to download the raw WMT14 data. The builder creates a `TranslationDataset` object.
2.  **Persistence**: The resulting "raw" `TranslationDataset` is added to the `DatasetRepo` (`repo.py`). The repository saves the manifest and the `.jsonl` data file to disk.
3.  **Filtering**: A user then retrieves the raw dataset from the repository and passes it to the `TranslationDatasetBuilderFiltered` (`data_filter.py`). This builder processes the data, applies filters, and creates a new "filtered" `TranslationDataset`.
4.  **Persistence (Again)**: This new, filtered dataset is also added to the `DatasetRepo`, which saves its unique manifest and data file.
5.  **Tokenization**: Finally, the filtered training dataset is retrieved from the repository and used by `BPETokenization` (`bpe.py`) to train a new tokenizer. The trained tokenizer is then saved to disk as a separate artifact.

---

## Example Usage

The primary entry point for a user of this package is typically the `DatasetRepo`.

```python
from mini_transformer.container import MiniTransformerContainer

# Initialize the container to get access to services
container = MiniTransformerContainer()
container.init_resources()
container.wire(modules=[__name__])

# Get the dataset repository
repo = container.data.repo()

# List available filtered datasets
filtered_datasets = repo.show(stage="filtered")
print(filtered_datasets)

# Retrieve a specific filtered training dataset
train_dataset_id = filtered_datasets.loc[
    (filtered_datasets["split"] == "train")
]["id"].values[0]

# Get the full, materialized dataset
training_data = repo.get(dataset_id=train_dataset_id)

# Now you can use the training_data object
print(f"Loaded dataset with {len(training_data)} examples.")