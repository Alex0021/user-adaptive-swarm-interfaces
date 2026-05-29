# User-Adaptive Swarm Interface through Eye-Tracking
*Robotics Master Thesis - EPFL*

- Eye-tracking models for cognitive load estimation of a user
- Data analysis pipeline


## Installation

I suggest using the python ```uv``` ([see here](https://docs.astral.sh/uv/getting-started/installation)) package manager as it allows to install speciific python versions easily.

1. Clone the git repo

```bash
git clone https://github.com/Alex0021/thesis-adaptive-swarm-interfaces.git 
```

2. Create the ```venv``` inside the *workload_inference* service
```bash
cd services/workload_inference
uv venv .venv --python=3.14
./.venv/Scripts/activate
```
> Note: the python version is specified as 3.14 here. The activation script might be under a slightly diffferent location for non windows users

3. Install the required packages
```bash
uv sync
uv pip install -e .
```

> Note: Make sure to use the ```-e``` (editable) mode, otherwise the relative path resolution won't work (should be improved at some point)

## Usage
The ```workload_inference``` command will launch the different experiment interfaces

```bash
workload_inference --experiment [nback|gates]
```

The ```plot_results``` is used to generate the different plots for analysis
```bash
plot_results --help
```

Other scripts are also available, just have a look at the ```pyproject.toml``` file