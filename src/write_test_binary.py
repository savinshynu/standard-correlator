import os
import sys
import argparse
import h5py
import numpy as np


class WriteSampleBinary:
    
    def __init__(self, infile, nchunks, outfile="."):
        self.file_path = infile
        self.outfile_path = outfile
        self.nchunks = nchunks # no. of integrations to write
        self.data = None
        self.load_header() # loading header from the DADA file

        
    def load_header(self):

        # Read header
        with open(self.file_path, 'rb') as f:
            header = f.read(4096).decode('ascii')
        self.header = self.parse_header(header)

    @staticmethod
    def parse_header(header):
        header_dict = {}
        for line in header.split('\n'):
            if line and not line.startswith('#') and ' ' in line:
                key, value = line.split(None, 1)
                header_dict[key] = value.strip()


        header_dict['NBIT'] = int(header_dict['NBIT'])
        header_dict['NDIM'] = int(header_dict['NDIM'])
        header_dict['NPOL'] = int(header_dict['NPOL'])
        header_dict['NCHAN'] = int(header_dict['NCHAN'])
        header_dict['NANT'] = int(header_dict['NANT'])
        header_dict['INNER_T'] = int(header_dict['INNER_T'])
        header_dict['HDR_SIZE'] = int(header_dict['HDR_SIZE'])
        header_dict['ORDER'] = str(header_dict['ORDER'])
        header_dict['CHAN0_IDX'] = int(header_dict['CHAN0_IDX'])
        header_dict['CHAN_WIDTH'] = float(header_dict['OBS_BW'])/float(header_dict['OBS_NCHAN'])
        header_dict['FBEG'] = float(header_dict['FREQ']) 
        return header_dict
    
    def write_data(self, int_dur=0.02):

        """
        Here we collect the time series data from all antennas and do the cross correlations of all antennas
        and save the corresponding visibility matrix, other data  and metadata needed for MS format.
        """
        
        raw_size = int(self.header['FILE_SIZE']) - int(self.header['HDR_SIZE'])
        #free_memory = psutil.virtual_memory()[4]/5

        if self.header['ORDER'] == 'TAFTP': #if this is not present is this a norm?
            # Read data
            with open(self.file_path, 'rb') as f:
                dp =  self.header['NANT']*self.header['NCHAN']*self.header['INNER_T']*self.header['NPOL']*self.header['NDIM'] # Minimum samples needed for reordering

                check = raw_size % dp
                if check != 0:
                    sys.exit("Check the order of data")
                
                nant = self.header['NANT']
                nchan = self.header['NCHAN']
                npol = self.header['NPOL']
                ndim = self.header['NDIM']
                inner_t = self.header['INNER_T']
                outer_t = int(int_dur/(self.header['INNER_T']*float(self.header['TSAMP'])*1e-6)) # Number of outer time steps to read at a time for an integration time

                nint = int(raw_size/(dp*outer_t)) # number of integrated time samples

                if self.nchunks < nint:
                    data = np.fromfile(f, dtype=np.int8, count=(self.header['HDR_SIZE']+dp*outer_t*self.nchunks)) #reading a portion of required data into the memory
                    print("Data acquired")
                    with open(self.outfile_path+"/test-big.dada", "wb") as fw:
                        fw.write(data)
                        print("Test file written to")

        else:
            sys.exit("Unknown data order for Meerkat")

                
if __name__ == "__main__":
    input_file = sys.argv[1]
    outfile_path = sys.argv[2]
    nchunks = 100

    wob = WriteSampleBinary(input_file, nchunks, outfile_path)
    wob.write_data()