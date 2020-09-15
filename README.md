# 4castr

## Set up and installation

To run, create a conda environment

```bash
conda create -n datares_4castr python==3.7
```

Then clone the repo

```bash
git clone https://github.com/datares/4castr.git
```

Finally, update your conda env with the specs that are necessary to run the project

```bash
cd 4castr
conda env update --name datares_4castr --file environment_root.yml
```

## Running the project

To run the project with optuna, run:

```bash
conda activate datares_4castr
python run.py -m optimize -n your_model_name
```
