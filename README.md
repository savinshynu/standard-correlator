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

Usage: ```python correlator.py [DADA file] [meta file] -o [output filepath] -ms (CASA MS if needed)```
Make sure to input the right input file by matching the date on the meta file and DADA file.