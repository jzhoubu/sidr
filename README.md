# SiDR: Semi-Parametric Retrieval via Binary Bag-of-Tokens Index


<p align="center">
  <img src="docs/images/train.png" width="55%" alt="Overview of SiDR Training">
</p>

## Installation
```
# install poetry first
# curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
```

## Quick Start
```python
# query: str or List[str]
# passages: List[str]
import torch
from src.ir import Retriever

query = "Who first proposed the theory of relativity?"
passages = [
    "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity.",
    "Sir Isaac Newton FRS (25 December 1642 – 20 March 1727) was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.",
    "Nikola Tesla (10 July 1856 – 7 January 1943) was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system."
]

# Load retriever
sidr = Retriever.from_pretrained("jzhoubu/sidr-nq") # sidr-nq (train on NQ) or sidr-ms (train on MSMARCO)
sidr = sidr.to("cuda")

# Embed query and passages
q_emb = sidr.encoder_q.embed(query)  # Shape: [1, V]
p_emb = sidr.encoder_p.embed(passages)  # Shape: [4, V]

# Relevance
scores = q_emb @ p_emb.t()
print(scores)

# Output: 
# tensor([[97.2964, 39.7844, 37.6955]], device='cuda:0')
```



## Build Index (for large-scale search)

```python
sidr.build_index(passages, index_type="sparse") # sparse embedding-based index
print(sidr.index)

# Output:
# Index Type      : SparseIndex
# Vector Shape    : torch.Size([3, 29523])
# Vector Dtype    : torch.float32
# Vector Layout   : torch.sparse_csr
# Number of Texts : 3
# Vector Device   : cuda:0

sidr.build_index(passages, index_type="bag_of_token") # bag-of-tokens index
print(sidr.index)

# Output:
# Index Type      : BoTIndex
# Vector Shape    : torch.Size([3, 29523])
# Vector Dtype    : torch.float16
# Vector Layout   : torch.sparse_csr
# Number of Texts : 3
# Vector Device   : cuda:0


# Save index
index_file = "/path/to/index.npz"
sidr.save_index(path)

# Load index
index_file = "/path/to/index.npz"
data_file = "/path/to/texts.jsonl"
sidr.load_index(index_file=index_file, data_file=data_file)
```

## Search on Index

<p align="center">
  <img src="docs/images/infer.png" width="75%" alt="Overview of SiDR Search">
<img src="docs/images/cost-effective.png" width="55%" alt="Cost-effectiveness">

</p>

### 1. Conventional Search (Embedded Query Search Embedded Passages)
```python
# Large-scale search on index
index_file=/path/to/embedding_based_index
sidr.load_index(index_file=index_file, data_file=data_file)

queries = [query]
results = sidr.retrieve(queries, k=3)
print(results)

# Output:
# SearchResults(
#   ids=tensor([[0, 1, 2]], device='cuda:0'),
#   scores=tensor([[97.2458, 39.7507, 37.6407]], device='cuda:0')
# )

# Get raw passage
query_id = 0
top1_psg_id = results.ids[query_id][0]
top1_psg = sidr.index.get_sample(top1_psg_id)
print(top1_psg)

# Output:
# Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity.
```

### 2. Beta Search (Embedded Query Search Tokenized Passages)
```python
# Small-scale beta search
q_emb = sidr.encoder_q.embed(query)
p_bin = sidr.encoder_p.embed(passages, bow=True)
scores = q_emb @ p_bin.t()
```

```python
# Large-scale search on index
index_file=/path/to/bag_of_tokens_index
sidr.load_index(index_file=index_file, data_file=data_file)

# Beta search
queries = [query]
beta_results = sidr.retrieve(queries, k=3)

# Beta search + Re-rank (late parametric)
queries = [query]
beta_rerank_results = sidr.retrieve(queries, k=3, rerank=True)
print(beta_rerank_results)

# Output:
# SearchResults(
#   ids=tensor([0, 2, 1], device='cuda:3'), 
#   scores=tensor([97.2964, 39.7844, 37.6955], device='cuda:0')
# )
```

## Citation
If you find this repository useful, please consider giving ⭐ and citing our paper:
```
@inproceedings{zhousemi,
  title={Semi-Parametric Retrieval via Binary Bag-of-Tokens Index},
  author={Zhou, Jiawei and Dong, Li and Wei, Furu and Chen, Lei},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```