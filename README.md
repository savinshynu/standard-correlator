### DZA Correlator for Meerkat
A python correlator for cross correlating the per antenna voltages from the Meerkat radio telescope in the DADA format.
Ouputs visibilities (XX, YY) in the UVh5 and CASA MS formats.

Mainly consist of 3 parts:
1. correlator.py - Correlator class reading the chunks of Meerkat voltage files and computing the correlated visibilities.
2. uvh5_tools.py - Function to collect all the data and metadata in the correct format to further write them into a UVH5 format (consisting of header and datasets).
3. compute_uvw.py - Function to calculate the UVW coordinates for each antenna baseline.

Dependencies required:
- astropy
- h5py
- pyuvdata
- katpoint
- numpy
- tqdm
- jax[CUDA]
- numba


## Usage

```sh
python correlator.py DADAfile METAfile [options]
```

**Positional arguments:**

- `DADAfile`: Input voltage data file in the `.dada` format  
- `METAfile`: Input metafile for the observations in `.hdf5` format

**Optional arguments:**

- `-o`, `--outdir`: Output directory for visibilities (default: current directory)
- `-ms`, `--casa_ms`: Output visibilities in CASA MS format (in addition to UVH5)
- `-r`, `--rem_uvh5`: Remove the UVH5 file after the conversion to CASA MS format
- `-b`, `--backend`: **Select backend for correlation:**
    - `cpu`: Use CPU with Numba acceleration
    - `gpu`: Use GPU with JAX (requires compatible GPU and JAX installation)  
    - *(default: `cpu`)*


## Example Commands

**Run on CPU using all cores, save output to `./results` as UVH5:**
```sh
python correlator.py test.dada test_meta.h5 -o ./results
```

**Run on GPU, save both UVH5 and CASA MS output, remove UVH5 after conversion:**
```sh
python correlator.py test.dada test_meta.h5 -o ./results -ms -r -b gpu
```
