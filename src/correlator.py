import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib
from katpoint import Antenna # Meerkat library for reading metafile
from tqdm import tqdm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from pyuvdata import UVData

from compute_uvw import meerkat_uvw
from uvh5_tools import create_uvh5



class Correlator:
    def __init__(self, file_path, meta_file_path):
        self.file_path = file_path
        self.meta_file = meta_file_path
        self.header = None
        self.data = None
        self.meta = {} # extracted information from the meta file
        self.load_header() # loading header from the DADA file
        self.extract_meta() # loading metadata from the observation metadata
        self.load_all_data() # load visibilities, UVW coordinates, flag, sample ratio, baseline information, plus add additional info to self.meta
        

    def load_header(self):

        # Read header
        with open(self.file_path, 'rb') as f:
            header = f.read(4096).decode('ascii')
        self.header = self.parse_header(header)


    def load_all_data(self, int_dur=0.1):

        """
        Here we collect the time series data from all antennas and do the cross correlations of all antennas
        and save the corresponding visibility matrix, other data  and metadata needed for MS format.
        """
        
        raw_size = int(self.header['FILE_SIZE']) - int(self.header['HDR_SIZE'])
        #free_memory = psutil.virtual_memory()[4]/5

        if self.header['ORDER'] == 'TAFTP': #if this is not present is this a norm?
            # Read data
            with open(self.file_path, 'rb') as f:
                f.seek(self.header['HDR_SIZE'])
                
                dp =  self.header['NANT']*self.header['NCHAN']*self.header['INNER_T']*self.header['NPOL']*self.header['NDIM'] # Minimum samples needed for reordering

                check = raw_size % dp
                if check != 0:
                    sys.exit("Check the order of data")
                
                nant = self.header['NANT']
                nbls = int(nant*(nant+1)/2) # correlations and autocorrelations
                nchan = self.header['NCHAN']
                npol = self.header['NPOL']
                ndim = self.header['NDIM']
                npol_prod = 2 #just consider RR and LL for now,  # output polarization products [RR*, RL*, LR*, LL*]
                inner_t = self.header['INNER_T']
                outer_t = int(int_dur/(self.header['INNER_T']*float(self.header['TSAMP'])*1e-6)) # Number of outer time steps to read at a time

                nint = int(raw_size/(dp*outer_t)) # number of integrated time samples
                
                # temporary setting for now 
                #nint = 2    
                
                ant_names_str = list(self.meta['antenna_positions'].keys()) # antenna names in the string format
                
                ant_names  = [int(ant[1:]) for ant in ant_names_str]   # antenna numbers without "m" in front

                ant1_array = np.zeros((nint, nbls), dtype = 'int32') # array for storing the baseline information 
                ant2_array = np.zeros((nint, nbls), dtype = 'int32')
        
                time_array = float(self.header['UTC_START']) + int_dur/2.0 + (np.arange(nint)*int_dur) # time array in unix time stamps (s), selecting points in the middle of integrations
                
                uvw_array = np.zeros((nint, nbls, 3), dtype = 'float32') #initialize the uvw array

                antpos = np.array(list(self.meta['antenna_positions'].values())) # Actual X, Y, Z antenna positions in ECEF (m), corresponding changes made while computing UVW. 
        
                # The pointing information in ra, dec strings to radians
                pointing = self.convert_dir2float(self.header['RA'], self.header['DEC'])
                
                # Adding extracted information into meta dict
                self.meta['nTimesteps'] = nint 
                self.meta['ant_index'] = ant_names 
                self.meta['ant_names_str'] = ant_names_str
                self.meta['time_array'] = time_array
                self.meta['pointing'] = pointing
                self.meta['tInt'] = int_dur
                
                # Defining array to store the visibilities, flag and sample ration per integration
                vis_mat = np.zeros((nint, nbls, nchan, npol), dtype='complex64')
                flag_mat = np.zeros(vis_mat.shape, dtype = 'bool') # flag information in the data
                nsamples_mat = np.ones(vis_mat.shape, dtype = 'float32') # fraction of samples going into each integration
                
                print("Reading chunks of data from the DADA files and cross correlating to get the visibility matrix")
                for num in tqdm(range(nint)):
                    chunk = np.fromfile(f, dtype=np.int8, count=dp*outer_t) #reading a portion of data into the memory

                    # get the UVW value at this time for all the antennas
                    uvw_now = meerkat_uvw(time_array[num], pointing, antpos)
                     
                    if chunk.size < dp*outer_t:
                        samp_ratio = round(chunk.size/(dp*outer_t), 3)
                        nsamples_mat[num,:, :, :] = samp_ratio
                    
                    #first reading based on how data is stored
                    chunk = np.reshape(chunk, (outer_t, nant, nchan, 
                            inner_t, npol, ndim)) 
            
                    # transposing to array the combine the outer and inner time axis
                    chunk = np.transpose(chunk, axes=(1,2,0,3,4,5)).reshape((nant, nchan, outer_t*inner_t, npol, ndim))

                    # converting that to a complex format
                    chunk = np.asarray(chunk, dtype='float32').view('complex64').squeeze()
                    #print(chunk.shape)

                    # calculate averaged visibilities 
                    vis_int, uvw_int, ant1_int, ant2_int = self.calc_vis_uvw_ant(chunk, uvw_now, ant_names)
                    
                    vis_mat[num, :, :, :] = vis_int
                    uvw_array[num, :, :] = uvw_int
                    ant1_array[num, :] = ant1_int
                    ant2_array[num, :] = ant2_int

                    num +=1 
                #reshaping all the array into nint*nbls format suitable for UVH5 datasets
                self.data = (vis_mat.reshape(nint*nbls, nchan, npol), uvw_array.reshape(nint*nbls,3), ant1_array.reshape(nint*nbls),
                            ant2_array.reshape(nint*nbls), flag_mat.reshape(nint*nbls, nchan, npol), nsamples_mat.reshape(nint*nbls, nchan, npol))

        
        else:
            sys.exit("Unknown data order for Meerkat")

    @staticmethod
    def calc_vis_uvw_ant(chunk, uvw_now, ant_names):
        """
        Calculate the visibility for each chunk read into the memory, UVW coordinates
        and collect baseline information.
        """
        
        nant, nchan, ntimes, _ = chunk.shape
        nprod = 2 # 2 polarization product for now
        nbls = int(nant*(nant+1)/2)
        vis_chunk = np.zeros((nbls, nchan, nprod), dtype='complex64')
        ant1_chunk = np.zeros((nbls), dtype = 'int32') # array for storing the baseline information 
        ant2_chunk = np.zeros((nbls), dtype = 'int32')
        uvw_chunk = np.zeros((nbls, 3), dtype = 'float32') #initialize the uvw array
        
        bls_ind = 0 # baseline index

        # Write out the auto correlations first
        for ant in range(nant):
            vis_chunk[bls_ind, :, 0] = (chunk[ant, :, :, 0] * np.conjugate(chunk[ant, :, :, 0])).mean(axis=1) # RR
            vis_chunk[bls_ind, :, 1] = (chunk[ant, :, :, 1] * np.conjugate(chunk[ant, :, :, 1])).mean(axis=1) # LL

            ant1_chunk[bls_ind] = ant_names[ant] # First antenna
            ant2_chunk[bls_ind] = ant_names[ant] # second antenna
            uvw_chunk[bls_ind,:] = 0.0 # uvw_ant1 - uvw_ant2 == 0 for autocorrelation

            bls_ind +=1 
        
        # write out the cross correlations now
        for ant1 in range(nant):
            if (ant1 + 1) < nant:
                for ant2 in range(ant1 +1 , nant):
                    vis_chunk[bls_ind, :, 0] = (chunk[ant1, :, :, 0] * np.conjugate(chunk[ant2, :, :, 0])).mean(axis=1) # XX
                    vis_chunk[bls_ind, :, 1] = (chunk[ant1, :, :, 1] * np.conjugate(chunk[ant2, :, :, 1])).mean(axis=1) # YY  

                    ant1_chunk[bls_ind] = ant_names[ant1]
                    ant2_chunk[bls_ind] = ant_names[ant2]
                    uvw_chunk[bls_ind,:] = uvw_now[ant1,:] - uvw_now[ant2,:] # difference in uvw coordinates between antenna 1 and 2

                    bls_ind += 1

        return (vis_chunk, uvw_chunk, ant1_chunk, ant2_chunk)
    
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
    
    @staticmethod
    def ant2bls(ant1, ant2):
    
        """
        Convert antenna index to baseline indec
        """
        
        (a1, a2) = sorted((ant1, ant2))

        #does not work when ant1 == ant2

        return (a2*(a2-1))//2 + a1
    
    @staticmethod
    def convert_dir2float(ra, dec):
        """
        Convert the ra and dec string into 
        radians
        """
        c = SkyCoord(ra+ ' ' + dec, unit=(u.hourangle, u.deg)) # Calling astropy to convert ra, dec string to radians
        return (c.ra.radian, c.dec.radian) # returning ra, dec tuple in radians


    def extract_meta(self):
        """
        Extract all important information from the metafile
        """
        ant_pos = {} # ECEF coordinates of each antenna elements
        with h5py.File(self.meta_file) as hf:
            
            # Parse out antenna to F-engine mapping, {needed to understand the actively data collecting antennas}
            antenna_feng_map = {antenna.decode(): index for antenna, index in hf["antenna_feng_map"][()]}
            
            # parse out antenna information
            for ant_info in hf["antenna_positions"]:
                # get an instance of Antenna class
                ant_ob = Antenna(ant_info.decode())
                if ant_ob.name in antenna_feng_map.keys():
                    ant_pos[ant_ob.name] = ant_ob.position_ecef # assign the corresponding ECEF coordinates in tuples

        
            self.meta["antenna_positions"] = ant_pos
            self.meta["antenna_feng_map"] = antenna_feng_map
            self.meta.update(dict(hf.attrs))

    def get_header_data(self):

        lat_mkat = -30.700184259 # latitude degrees 
        lon_mkat =  21.433509259 # longitude degrees
        alt_mkat = 1086.6  # altitude in meters
        tel_name = self.header['TELESCOPE'] # telescope name
        instrument = self.header['INSTRUMENT'] # instrument name
        history = "Ask savin"
        nants_data = self.header['NANT'] # antennas present in the data
        nants_tel = 64 # antennas present in the telescope
        nbls = int(nants_data*(nants_data+1)/2.0) # number of baselines
        ntimes = self.meta['nTimesteps'] # time samples
        nbltimes = nbls*ntimes # baseline * ntimes
        nfreqs = self.header['NCHAN'] # No. of frequency channels
        nspws = 1 # spectral windows
        npols = 2 # polarization products, ideally 4, setting 2 now for RR and LL

        ## collecting important data
        vis_data, uvw_array, ant1_array, ant2_array, flag_data, nsamples_data = self.data 

        # antenna numbers corresponding to each baseline-time pair
        #ant_names = np.array([b'ea01', b'ea02', b'ea03', b'ea04', b'ea05', b'ea06', b'ea07', b'ea08', b'ea09', b'ea10', b'ea11', b'ea12', b'ea13', b'ea14', b'ea15',
        # b'ea16', b'ea17', b'ea18', b'ea19', b'ea20', b'ea21', b'ea22', b'ea23', b'ea24', b'ea25', b'ea26', b'ea27', b'ea28'], dtype = 'object')
        #ant_numbers = np.arange(1,29, dtype = 'int32') # antenna numbers without  ea present in the telescope
        
        ant_names = [name.encode() for name in self.meta['ant_names_str']]
        ant_numbers = self.meta['ant_index']

        freq_array = self.header['FBEG'] + np.arange(self.header['NCHAN'])*self.header['CHAN_WIDTH'] # frequency array in Hz
        chan_width = self.header['CHAN_WIDTH']*np.ones(self.header['NCHAN']) # channel width in Hz
        antenna_diameter = np.ones(self.header['NANT'])*13.5 # antenna diameter of each dish, for meetkat = 13.5 m
        antenna_positions = self.get_antenna_positions_ref() # ECEF coordinates relative to the reference position of the array
        integration_time = np.ones(nbltimes)*self.meta['tInt']
        #time_array_unix = self.stamp_meta['tstart'] + np.arange(ntimes)*self.stamp_meta['tsamp'] # creating time array for the stamp in UNIX format
        time_array_unix_astro = time.Time(self.meta['time_array'], format='unix') # loading unix time stamps to astropy
        time_array_jd = time_array_unix_astro.jd  # coverting time stamps to JD day format for UVH5
        time_bls_array = np.repeat(time_array_jd, nbls) # converting that to ntimes * nbaselines
        spw_array = np.ones((1), dtype = 'int32') #only one spectral index
        flex_spw = False # set to true if more than 1 spectral windows
        #pol_array = np.array([-1, -3, -4, -2], dtype='int32') # RR, RL, LR, LL [-1, -3, -4, -2]
        pol_array = np.array([-5, -6]) # XX, YY, XY, YX [-5, -6, -7, -8], currently only have XX and YY
        version = '1.0'.encode()
        object = self.header["SOURCE"].encode()
        phase_type = 'phased'.encode() # assuming the input data is phased
        phase_center_ra = self.meta['pointing'][0] # ra and dec n radians
        phase_center_dec = self.meta['pointing'][1]
        phase_center_epoch = 2000.0 # assuming coordinates are in J2000 epoch
        phase_center_frame = 'icrs'.encode()
        extra_keywords = ''

        #Other data
        #flag_data = np.zeros(visdata.shape, dtype = 'bool')
        #nsamples = np.ones(visdata.shape, dtype = 'float32')
        
        head_dict = {'Nants_data': nants_data  , 'Nants_telescope': nants_data, 'Nbls': nbls, 'Nblts': nbltimes, 'Nfreqs':nfreqs, 'Npols':npols, 'Nspws':nspws, 'Ntimes':ntimes,
         'altitude': alt_mkat, 'ant_1_array': ant1_array, 'ant_2_array': ant2_array, 'antenna_diameters': antenna_diameter, 'antenna_names': ant_names, 'antenna_numbers': ant_numbers, 
         'antenna_positions': antenna_positions, 'channel_width': chan_width, 'extra_keywords': extra_keywords, 'flex_spw': flex_spw, 'freq_array': freq_array,
         'history': history , 'instrument': instrument, 'integration_time': integration_time, 'latitude': lat_mkat, 'longitude': lon_mkat, 'object_name': object, 'phase_center_dec': phase_center_dec, 
         'phase_center_epoch': phase_center_epoch, 'phase_center_frame': phase_center_frame,  'phase_center_ra': phase_center_ra, 'phase_type': phase_type, 'polarization_array': pol_array,
          'spw_array': spw_array, 'telescope_name': tel_name, 'time_array': time_bls_array, 'uvw_array': uvw_array, 'version': version }
        
        data_dict = {'flags': flag_data, 'nsamples': nsamples_data, 'visdata': vis_data }
        #print(head_dict)
        return head_dict, data_dict
    
    def get_antenna_positions_ref(self, ref_ant=None):
        """
        Collect the antenna positions wrt to the reference antenna in the ECEF format
        If no reference antenna given, use the array center location
        """
        ant_pos = self.meta['antenna_positions'] # dictionary containing values
        
        if ref_ant:
            ref_ecef = ant_pos[ref_ant]
        else:
            # Use the the coordinates of the center of the array
            ref_ecef = (5109360.0, 2006852.5, -3238948.0)
        
        ant_pos_ecef = np.array(list(self.meta['antenna_positions'].values())) # Actual X, Y, Z antenna positions in ECEF (m)
        return (ant_pos_ecef - np.array(ref_ecef)) # Antenna positions in XYZ wrt to reference antenna or center of the array

    def write_uvh5(self, outpath = '.', msdata=False):
        """
        Write the header and data into a uvh5 file
        """
        
        filepath_uvh5 = os.path.join(outpath, os.path.splitext(os.path.basename(self.file_path))[0]+".uvh5")
        print(f"Writing out {filepath_uvh5}")
        fob = h5py.File(filepath_uvh5, "w") # creating the uvh5 file
        head_dict, data_dict = self.get_header_data() # collecting all the important data and header
        create_uvh5(fob, head_dict, data_dict) # Writing all the data into the uvh5 file handle
        fob.close() # close afer after writing

        if msdata: # if needed to convert the UVH5 data into the CASA MS format
            print("Writing out the MF format file")
            uvd = UVData()
            uvd.read(filepath_uvh5, fix_old_proj=False)
            outfile_ms = os.path.join(outpath, os.path.splitext(os.path.basename(self.file_path))[0]+".ms")
            uvd.write_ms(outfile_ms)

def main(args):
    
    fob = Correlator(args.DADAfile, args.METAfile)
    print(fob.header)
    
    #vis_mat, uvw_array, ant1_array, ant2_array, flag_mat, nsamples_mat = fob.data
    fob.write_uvh5(outpath="full", msdata=True)
    
    


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description="For reading and visualizing DADA files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('DADAfile')
    parser.add_argument('METAfile')
    args = parser.parse_args()
    
    # run the main function
    main(args)
