# From One Network to Another

This is the corresponding artifact to the paper `From One Network to Another: Transfer Active Learning for Domain Adaptation of Flow Classifiers` to `AINA 2026`.

## Setup

We use [poetry](https://python-poetry.org/) for managing dependencies.
Install the dependencies by running:

```bash
# Install dependencies and virtual environment
$ poetry install
```

## Repo Structure

The code is structured into two parts:

- The `shared` module contains general code, not directly related to flow-based classification.

- The `flow` module, which handles primary code for flow classification. The scripts to our primary experiments can be found in `flc/flows/experiments`.

### Create datasets

To prepare the datasets for our experiments, a directory with the PCAP files (already split into separate, bi-direction flows) is expected.

#### Annotate Flows (flc/flows/scripts/flow_labeling)

Create a `.yaml` file according to the configuration defined in `flc/flows/scripts/flow_labeling/config.py`.

Then call the annotation script:

```bash
$ python3 flc/flows/scripts/flow_labeling/run.py --config <path-to-config-yaml>
```

#### Create dataset (flc/flows/scripts/create_flow_classification_dataset)

Create a `.yaml` configuration file according to `flc/flows/scripts/create_flow_classification_dataset/config.py`.

Then call the script:

```bash
$ python3 flc/flows/scripts/create_flow_classification_dataset/run.py --config <path-to-config-yaml>
```

#### Split dataset into train/valid/test (flc/flows/scripts/create_dataset_splits)

Create a `.yaml` configuration file according to `flc/flows/scripts/create_dataset_splits/config.py`.

Then call the script:

```bash
$ python3 flc/flows/scripts/create_dataset_splits/run.py --config <path-to-config-yaml>
```
