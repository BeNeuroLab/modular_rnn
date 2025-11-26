# modular_rnn

A small package for quick prototyping of multi-region (or modular) recurrent neural network (RNN) models on neuroscience tasks.

## Installation
Install in an existing environment:
```
git clone https://github.com/BeNeuroLab/modular_rnn.git
cd modular_rnn
pip install -e .
```

## Example use
For examples see notebooks modeling the [classic center-out reaching task](center_out_example.ipynb) and the ["uncertainty task"](uncertainty_task_example.ipynb) [from Dekleva et al. 2016](https://elifesciences.org/articles/14316).

To just quickly play around with the examples, you can use uv:
1. Install uv following the [instructions](https://docs.astral.sh/uv/getting-started/installation/#installation-methods), which on Linux or MacOS is just:

   `curl -LsSf https://astral.sh/uv/install.sh | sh`
   
4. `uv run --extra examples jupyter lab center_out_example.ipynb`
