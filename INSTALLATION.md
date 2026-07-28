# Installation

This is a recipe-style installation guide for Linux. It assumes that you have
already cloned the `alaric` repository and are running commands from the
repository root.

## 1. Obtain the `attract-jax` submodule

Initialize and fetch the `attract-jax` submodule:

```bash
git submodule update --init --recursive attract-jax
```

If you are cloning `alaric` from scratch, you can also fetch submodules during
the clone:

```bash
git clone --recurse-submodules <ALARIC_REPOSITORY_URL>
cd alaric
```

## 2. Install Miniforge3

Download and install Miniforge3:

```bash
wget -O Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Follow the installer prompts. After installation, either open a new shell or
initialize conda in the current one:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

If you installed Miniforge somewhere other than `~/miniforge3`, adjust the path
above.

## 3. Create the `alaric` conda environment

Create a fresh conda environment with Python 3.12:

```bash
conda create -n alaric python=3.12 pip
conda activate alaric
```

## 4. Install `attract-jax` first

`attract-jax` must be installed before `alaric`.

From the `alaric` repository root:

```bash
pip install -e attract-jax
```

## 5. Install `alaric`

With the same `alaric` conda environment active, install `alaric` in editable
mode:

```bash
pip install -e .
```

You can check that the command-line entry points are available with:

```bash
alaric-chain --help
alaric-deploy --help
```

## 6. Repeat the installation on the remote machine

If you use Alaric's remote execution or deployment workflow, repeat the same
installation steps on the remote machine:

```bash
ssh <REMOTE_HOST>
cd ~/alaric
git submodule update --init --recursive attract-jax
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alaric
pip install -e attract-jax
pip install -e .
```

The remote machine needs its own working checkout, Miniforge installation,
conda environment, and editable installs. Installing locally is not enough,
because remote jobs execute in the remote filesystem and remote Python
environment.

## 7. Configure environment variables

Set the local and remote paths used by Alaric. For example:

```bash
export ALARIC_DIR=~/alaric/alaric
export ALARIC_REMOTE_HOST=mbi-frontend
export ALARIC_REMOTE_DEPLOYMENT_DIR=/users/sdevries/alaric-deployment
export ALARIC_REMOTE_ALARIC_DIR=/users/sdevries/alaric/alaric
export ALARIC_REMOTE_RESULT_DIR=/data3/sdevries/alaric-results
```

Meaning:

- `ALARIC_DIR`: local path to the `alaric/` Python package directory.
- `ALARIC_REMOTE_HOST`: SSH host name for the remote machine.
- `ALARIC_REMOTE_DEPLOYMENT_DIR`: remote directory where deployment files are
  written.
- `ALARIC_REMOTE_ALARIC_DIR`: remote path to the `alaric/` Python package
  directory.
- `ALARIC_REMOTE_RESULT_DIR`: remote directory where results are stored.

To make these settings persistent, add them to your shell startup file:

```bash
nano ~/.bashrc
```

Then paste the `export` commands, save the file, and reload it:

```bash
source ~/.bashrc
```

If you use a different shell, put the same `export` commands in that shell's
startup file.

## 8. Configure the fragment library

Alaric also reads fragment-library paths from:

```bash
~/.alaric/fraglib.yaml
```

Create the configuration directory:

```bash
mkdir -p ~/.alaric
```

Then create `~/.alaric/fraglib.yaml`. Use paths that match where the fragment
library is installed on your machine.

Example file:

```
CONFORMER_DIR: /opt/fraglib/library
ROTAMER_DIR: /opt/fraglib/rotaconformers
CRMSD_DIR: /opt/fraglib/crmsd

fraglen: 2
conformers: $CONFORMER_DIR/dinuc-XX-0.5.npy
conformer_replacements: $CONFORMER_DIR/dinuc-XX-0.5-replacement.npy
conformer_replacement_origins: $CONFORMER_DIR/dinuc-XX-0.5-replacement.txt
conformer_extensions: $CONFORMER_DIR/dinuc-XX-0.5-extension.npy
conformer_extension_origins: $CONFORMER_DIR/dinuc-XX-0.5-extension.origin.txt
rotamers: $ROTAMER_DIR/dinuc-XX-0.5.npy
rotamers_indices: $ROTAMER_DIR/dinuc-XX-0.5.index.npy
rotamer_extensions: $ROTAMER_DIR/dinuc-XX-0.5-extension.npy
rotamer_extension_indices: $ROTAMER_DIR/dinuc-XX-0.5-extension.index.npy
crmsds: $CRMSD_DIR/crmsd_matrix_XXX.npy
```

The `XX` and `XXX` strings are placeholders used by Alaric. Leave them exactly
as shown. At runtime, Alaric replaces `XX` with a dinucleotide sequence such as
`AA`, `AC`, or `GU`, and replaces `XXX` with a trinucleotide sequence for cRMSD
lookups.

The file may use `~`, environment variables, or variables defined earlier in the
same YAML file.
This file must exist on any machine that runs Alaric code. If you use remote
execution, create `~/.alaric/fraglib.yaml` on the remote machine too, with paths
that are valid from the remote machine.

## 9. Quick verification

In a fresh shell:

```bash
conda activate alaric
alaric-chain --help
```