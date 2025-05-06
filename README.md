### DZA Correlator for Meerkat
A python correlator for cross correlating the per antenna voltages from the Meerkat radio telescope in the DADA format.

Mainly consist of 3 parts:
1. correlator.py - Correlator class reading the chunks of Meerkat voltage files and computing the correlated visibilities.
2. uvh5_tools.py - Function to collect all the data and metadata in the correct format to further write them into a UVH5 format (consisting of header and datasets).
3. compute_uvw.py - Function to calculate the UVW coordinates for each antenna baseline.