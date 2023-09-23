# Lumorph-CL: Lumorph Collective Library

Lumorph-CL accepts command-line arguments for the number of GPUs and the collective algorithm to use and the model architecture. It generates the collective schedule times that is fed into flexlflow for simulation. The script requires a list of LayerToLayer and AllReduce calls of the model which is pre-populated in ```./logs/{model}``` directory. The output is a set of two files one which is passed as input to FlexFlow for simulation and another is a json file which contains the MCF routes in each stage of every unique all-reduce and L2L call. They will be available in the same ```./logs/{model}``` once the script completes.

## Requirements

```gurobipy```
```networkx```

## Usage

```python
python taskgraph_schedule.py --gpus <number of GPUs> --algorithm <algorithm to use> --model <model to use>
```

## Arguments

The following arguments are available:

* `--gpus`: Number of GPUs. Required argument. Accepts values from a list of choices: `[4, 8 , 16, 32, 64, 128, 256]`.
* `--algorithm`: Algorithm to use. Required argument. Accepts values from a list of choices: `['lumorph-2', 'lumorph-4', 'ring']`.
* `--model`: Model to use. Required argument. Accepts values from a list of choices: `['bert', 'ncf', 'inception']`.

## Example

```python
python script.py --gpus 8 --algorithm lumorph-2 --model bert
```
