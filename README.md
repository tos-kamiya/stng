# stng

The `stng` is an off-the-shelf grep-like tool that performs semantic similarity search.

**⚠️ ITS HIGHLY EXPERIMENTAL.**

## TL;DR (typical usage)

Search for the document files similar to the query phrase.

```sh
stng -v <query_phrase> <document_files>...
```

Example of search:  
![](docs/images/run1.png)

## Links

* Sentence-BERT https://www.sbert.net/

* Reimers, N., Gurevych, I., Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019. https://arxiv.org/abs/1908.10084

## History

#### 0.1.1

* fix: replace model with sentence-transformers/stsb-xlm-r-multilingual

#### 0.1.0

* First release
