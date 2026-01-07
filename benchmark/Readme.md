
# LlamaIndex Ingestion Benchmark (Config Set 1–5)
### Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com; Date: 06/01/2026

This benchmark implements and compares five ingestion configurations:

## Config Set 1: Default
- Sync load: `SimpleDirectoryReader(...).load_data()`
- Sync pipeline: `IngestionPipeline.run()` with no workers
- Embedding + upsert sequential (no batching)

## Config Set 2: Reader Parallel
- Parallel load: `SimpleDirectoryReader(...).load_data(num_workers=8)`
- Sync pipeline: `IngestionPipeline.run()` sequential
- Embedding + upsert sequential

## Config Set 3: Pipeline Parallel (Sync)
- Sync load
- Sync pipeline with multiprocessing: `IngestionPipeline.run(num_workers=4 or 8)`
- Embedding + upsert sequential (to isolate pipeline parallelism)

## Config Set 4: Async Only (concurrency)
- Sync load
- Async transforms: `await IngestionPipeline.arun(num_workers=N)`
- Async embedding with concurrency, but **no batching**:
  - `embed_batch_size = 1`
  - `upsert_batch_size = 1`

## Config Set 5: Async + Batching
- Sync load
- Async transforms
- Async embedding + batching + batched DB inserts:
  - `embed_batch_size = 64` (configurable)
  - `upsert_batch_size = 256` (configurable)

---

# Why this demonstrates Set4 vs Set5

Set 4 increases concurrency (more in-flight requests) but still pays per-request overhead for each node.
Set 5 keeps concurrency and additionally reduces overhead by batching embedding calls and vector DB inserts.

We simulate a realistic embedding API cost model:
- `latency = request_overhead_ms + per_item_ms * batch_size`

So batching reduces the number of requests and improves throughput.

---

# Dependencies

```bash
pip install llama-index chromadb

Run:
python benchmark_configs_1_to_5.py
```


It prints a table with:

```bash
load time

transform time

embedding time

upsert time

total time
```

## faster vs Set1 baseline
### Common experiments

```bash
Scale node count
python benchmark_configs_1_to_5.py --nodes 1000
python benchmark_configs_1_to_5.py --nodes 10000
python benchmark_configs_1_to_5.py --nodes 50000
```

### Change reader and pipeline workers
```bash
python benchmark_configs_1_to_5.py --reader-workers 8 --pipeline-workers 8
python benchmark_configs_1_to_5.py --reader-workers 16 --pipeline-workers 4
```

### Tune async concurrency
```bash
python benchmark_configs_1_to_5.py --async-workers 8
python benchmark_configs_1_to_5.py --async-workers 32
python benchmark_configs_1_to_5.py --async-workers 64
```

### Tune Set 5 batching
```bash
python benchmark_configs_1_to_5.py --set5-embed-batch 32 --set5-upsert-batch 128
python benchmark_configs_1_to_5.py --set5-embed-batch 128 --set5-upsert-batch 512
```

### Simulate different embedding API behaviors

```bash
python benchmark_configs_1_to_5.py --request-overhead-ms 80 --per-item-ms 1.0
python benchmark_configs_1_to_5.py --request-overhead-ms 10 --per-item-ms 2.0
```

## Notes / troubleshooting

If Set3 multiprocessing hangs or errors in your environment, try:

using fewer --pipeline-workers, or

switching to async ingestion patterns.
(Parallel processing relies on multiprocessing and can be sensitive to environment + transforms.)

This script uses a delimiter-based splitter to guarantee that the number of nodes equals --nodes.
The loader creates text files where each node-chunk is separated by a fixed delimiter.