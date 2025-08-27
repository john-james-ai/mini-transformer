# Infrastructure Package (`mini_transformer/infra`)

This package provides the foundational, low-level services that support the entire `mini-transformer` project. It is responsible for abstracting away details related to data storage, external data source interaction, and integration with the machine learning framework (PyTorch).

---

## Core Components

The package is divided into two main areas of responsibility: the Data Access Layer (`dal`) and Dataset I/O (`dataset`).

### 1. `dal/` (Data Access Layer)

This sub-package provides a generic and reusable abstraction for data persistence. It decouples the application logic from the specific storage implementation.

* **`oal.py` (Object Access Layer)**
    * **Purpose**: Provides a key-value store for arbitrary Python objects.
    * **Implementation**: Uses Python's built-in `shelve` module to create a simple, persistent dictionary-like database on disk.
    * **Use Case**: In this project, it is used by the `DatasetRepo` to store the lightweight "manifest" objects (the `TranslationDataset` instances with their data payload removed).

* **`fal.py` (File Access Layer)**
    * **Purpose**: Provides an interface for reading and writing bulk data to files.
    * **Implementation**: Stores data as `.jsonl` (JSON Lines) files, which is an efficient format for record-oriented data.
    * **Use Case**: Used by the `DatasetRepo` to store the actual data payload (the list of examples) for each dataset.

### 2. `dataset/` (Dataset I/O and Loading)

This sub-package contains the tools for fetching data from external sources and preparing it for consumption by PyTorch.

* **`download.py`**
    * **Purpose**: Handles the streaming of datasets from the Hugging Face Hub.
    * **Implementation**: The `HFDatasetDownloader` class uses the `datasets` library in `streaming=True` mode. This is highly memory-efficient as it processes the dataset as an iterable without downloading the entire file at once. It also supports shuffling and taking a subset of the stream.

* **`load.py`**
    * **Purpose**: Contains the necessary components to bridge our custom datasets with PyTorch's `DataLoader`.
    * **`PTDataset`**: A simple adapter class that wraps our `List[Dict]` data structure, making it compatible with the `torch.utils.data.Dataset` interface.
    * **`collate_fn`**: A critical function that defines how a list of individual examples (source and target strings) are batched together. It performs tokenization, padding, and masking on-the-fly to create tensors ready for the model.
    * **`BucketBatchSampler`**: A highly recommended sampler that groups sequences of similar lengths into the same batches. This minimizes the amount of padding required, leading to significantly more efficient training.

---

## PyTorch `DataLoader` Integration

The components in `load.py` are designed to work together to create an efficient data loading pipeline with PyTorch.

### Workflow with Bucketing

Using the `BucketBatchSampler` is a critical optimization. Instead of shuffling the entire dataset randomly, it groups examples into "buckets" based on their sequence length. Batches are then formed by drawing from these buckets, ensuring that the sequences within a single batch are of similar length. This dramatically reduces the amount of wasted computation on `<PAD>` tokens.

The workflow is as follows:

1.  A `TranslationDataset` is loaded from the `DatasetRepo`.
2.  The lengths of all source texts are extracted to be used by the sampler.
3.  The `BucketBatchSampler` is initialized with these lengths.
4.  The raw data (`dataset.data`) is wrapped in the `PTDataset` class.
5.  A `DataLoader` is instantiated, passing the `BucketBatchSampler` to the `batch_sampler` argument.
6.  The `collate_fn` is passed to the `DataLoader` to perform on-the-fly tokenization and tensor creation for each batch.

### Example Usage

Here is the recommended way to create a `DataLoader` using the `BucketBatchSampler` for efficient training:

```python
from functools import partial
from torch.utils.data import DataLoader
from mini_transformer.infra.dataset.load import PTDataset, collate_fn, BucketBatchSampler
from mini_transformer.data.tokenize.bpe import BPETokenization

# Assume 'repo' is an initialized DatasetRepo and a tokenizer has been trained

# 1. Load the filtered training data and the tokenizer
train_dataset_id = repo.show(stage="filtered", split="train")['id'].values[0]
train_dataset = repo.get(dataset_id=train_dataset_id)
tokenizer_service = BPETokenization(filepath="path/to/your/tokenizer.json")
tokenizer_service.load()
hf_tokenizer = tokenizer_service.tokenizer

# 2. Get the source sequence lengths for the bucket sampler
src_lengths = [len(item['src'].split()) for item in train_dataset.data]

# 3. Instantiate the BucketBatchSampler
bucket_sampler = BucketBatchSampler(
    bucket_ids=src_lengths,  # This will be converted to bucket IDs internally
    batch_size=32,
    len_key=src_lengths,     # Pass the actual lengths here
    drop_last=True           # Recommended for stable training
)

# 4. Wrap the data in the PyTorch-compatible Dataset
pytorch_dataset = PTDataset(data=train_dataset.data)

# 5. Create a partial function for the collate_fn to bind the tokenizer
collate_with_tokenizer = partial(collate_fn, tok=hf_tokenizer)

# 6. Instantiate the DataLoader using the BATCH_SAMPLER
#    IMPORTANT: When using batch_sampler, `batch_size`, `shuffle`,
#    `sampler`, and `drop_last` arguments must be None (the defaults).
train_dataloader = DataLoader(
    pytorch_dataset,
    batch_sampler=bucket_sampler,
    collate_fn=collate_with_tokenizer,
    num_workers=4  # Optional: for parallel data loading
)

# 7. Iterate over the DataLoader to get efficiently padded batches
for batch in train_dataloader:
    encoder_input = batch["encoder_input_ids"]
    print("Encoder Input Shape:", encoder_input.shape)
    # Example Output: Encoder Input Shape: torch.Size([32, 28])
    # Note how the sequence length is much smaller than the max (64)
    break