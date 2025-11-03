# SpecCache

This repository contains the data and code for paper [What Limits Agentic Systems Efficiency?](https://arxiv.org/pdf/2510.16276).

<p align="center">
    <a href="#benchmark">Benchmark 🛠️</a> •
    <a href="#agent">Agent ⚙️</a> •
    <a href="#contributing">Contributing 🐜</a> •
</p>

## Benchmark

The code used to measure LLM latency (as described in Section 2 of the paper) is provided in the `benchmark` folder.

### Environment Setup

To get started on benchmarking, please first setup the environment:
```bash
cd benchmark
conda create -n api python=3.10
pip install -r requirements.txt
```

### Configure API Key

Replace `OAI_API_KEY` in `utils_completions.py` and `utils_completions_priority.py` with your API key.

### Run Experiments

To run the standard LLM latency measurement: 
```bash
python benchmark/parallel/run.py
```
To enable OpenAI priority processing, run:
```bash
python benchmark/parallel_priority/run.py
```

To evaluate latency across different LLM providers, update the respective API key variables in `utils_completions.py`:
`DS_API_KEY` (DeepSeek), `TOGETHER_API_KEY` (Together AI), `ANTHROPIC_API_KEY` (Anthropic), `GOOGLE_API_KEY` (Google), and `CENTML_API_KEY` (CentML).


## Agent

### Environment Setup

SpecCache is implemented on top of [Qwen WebWalker Agent](https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent/WebWalker). To get started on running SpecCache Agent (as described in Section 3&4 of the paper), please set up the environment the following way:
```bash
cd SpecCache
conda create -n speccache python=3.10
pip install -r requirements.txt
crawl4ai-setup
crawl4ai-doctor
```

### Configure API Key

Replace `api_key`, `provider`, and `model_server` in `speccache_webwalkerqa_example.py` with your own setup.

### Run Experiments

To run the agent:
```bash
cd SpecCache
python speccache_webwalkerqa_example.py
```

Currently, the agent tests on the English subset (provided by the WebWalker_QA dataset), feel free to change the dataset and the dataset parsing code to deploy on a wider variety of datasets.

## Contributing

Authors: Song Bian*, Minghao Yan*, Anand Jayarajan, Gennady Pekhimenko, Shivaram Venkataraman

Affiliated: University of Wisconsin-Madison, University of Toronto and NVIDIA.

## Citation

If you find the idea or code useful for your research, please consider citing our [paper](https://arxiv.org/pdf/2510.16276):

```bib
@article{bian2025limits,
  title={What Limits Agentic Systems Efficiency?},
  author={Bian, Song and Yan, Minghao and Jayarajan, Anand and Pekhimenko, Gennady and Venkataraman, Shivaram},
  journal={arXiv preprint arXiv:2510.16276},
  year={2025}
}
```