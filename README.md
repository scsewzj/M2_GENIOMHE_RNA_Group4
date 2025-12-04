# Release V1 of Distance-dependent RNA Statistical Potential

Usually the statistical potential is derived from a large set of known RNA structures. 

This repository provides scripts to train, plot, and score RNA structures using distance-dependent statistical potentials. 

The workflow includes three main scripts under path `./scripts` : `training.py`, `plotting.py`, and `scoring.py`. The typical usage of these scripts is described below. 

Simply, first train the potential using known RNA structures, then plot the potential for visualization, and finally use the potential to score new RNA structures.

## Install
### Conda
You can use conda or any other variants package manager of python compatible with it to install(e.g.: mamba, micromamba etc.).
```bash
conda env create -f environment.yml -n rnascore
conda activate rnascore
```
### Docker
Under Development

## Training
Below is an example command for training the statistical potential using PDB/MMCIF files stored in `./data/pdbs/` directory. You can specify your own input file or directory and output directory as needed. Other options are provided including bin size, distance cutoff, etc. Use the `-h` flag to see all available options.
```bash
python ./scripts/training.py ./data/pdbs/
# python training.py <input_file/dir> -o <output_dir>
# python training.py -h
```


## Plotting
To visualize the statistical potential, you can use the following command. This will generate plots based on the computed potentials. Again, you can specify your own input and output directories as needed. Use the `-h` flag to see all available options.

### An example plot
![alt text](./potential_plot_sample.png)

```bash
python ./scripts/plotting.py -p ./potentials/ -o ./plots/
# python plotting.py -p <input_potential_dir> -o <output_plot_dir>
# python plotting.py -h
```

## Scoring
To score RNA structures using the computed statistical potentials, use the following command. You can specify your own input PDB/MMCIF files and output directory as needed. The output will be in the form of a CSV table. Use the `-h` flag to see all available options.

```bash
python ./scripts/scoring.py ./data/pdbs/ -p ./potentials/ -o ./scores.csv
# python scoring.py <input_pdb_dir> -p <input_potential_dir> -o <output_score_file>
# python scoring.py -h
```
