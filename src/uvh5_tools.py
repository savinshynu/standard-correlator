import h5py 

def create_uvh5(fob, head_dict, data_dict):
    """
    Create a UVH5 file containing correlated data
    Parameters:
    fob: h5py file handle
    outpath : Output file path
    head_dict : metadata dictionary
    data_dict: dictionary containing containing all the data
    """

    uvh5_header = fob.create_group("Header")
    uvh5_data = fob.create_group("Data")

    uvh5_header.create_dataset("longitude", data=head_dict['longitude'], dtype='d') # degrees
    uvh5_header.create_dataset("latitude", data=head_dict['latitude'], dtype='d') # degrees
    uvh5_header.create_dataset("altitude", data=head_dict['altitude'], dtype='d')

    uvh5_header.create_dataset("telescope_name", data=head_dict['telescope_name'])
    uvh5_header.create_dataset("instrument", data=head_dict['instrument'])
    uvh5_header.create_dataset("object_name", data=head_dict['object_name'])
    uvh5_header.create_dataset("history", data=head_dict['history'])
    uvh5_header.create_dataset("phase_type", data=head_dict['phase_type'])
    uvh5_header.create_dataset("Nants_data", data=head_dict['Nants_data'], dtype='i')
    uvh5_header.create_dataset("Nants_telescope", data=head_dict['Nants_telescope'], dtype='i')

    
    uvh5_header.create_dataset("antenna_names", data=head_dict['antenna_names'], dtype=h5py.special_dtype(vlen=str))
    uvh5_header.create_dataset("antenna_numbers", data=head_dict['antenna_numbers'], dtype='i')
    uvh5_header.create_dataset("antenna_diameters", data=head_dict['antenna_diameters'], dtype='d')
    uvh5_header.create_dataset("antenna_positions", data=head_dict['antenna_positions'], dtype='d')

    uvh5_header.create_dataset("Nbls", data=head_dict['Nbls'], dtype='i')
    uvh5_header.create_dataset("Nfreqs", data=head_dict['Nfreqs'], dtype='i')
    uvh5_header.create_dataset("Npols", data=head_dict['Npols'], dtype='i')
    uvh5_header.create_dataset("freq_array", data=head_dict['freq_array'], dtype='d')
    
    uvh5_header.create_dataset("channel_width", data=head_dict['channel_width'], dtype='d')
    uvh5_header.create_dataset("Nspws", data=head_dict['Nspws'], dtype='i')
    uvh5_header.create_dataset("spw_array", data=head_dict['spw_array'], dtype='i')
    uvh5_header.create_dataset("flex_spw", data=head_dict['flex_spw'])

    uvh5_header.create_dataset("polarization_array", data=head_dict['polarization_array'], dtype='i')

    uvh5_header.create_dataset("version", data=head_dict['version'])
    # uvh5_header.create_dataset("flex_spw_id_array", data=) # 1 int
    #uvh5_header.create_dataset("dut1", data=dut1, dtype='d')
    # uvh5_header.create_dataset("earth_omega", data=) # 0 double
    # uvh5_header.create_dataset("gst0", data=) # 0 double
    # uvh5_header.create_dataset("rdate", data=) # 0 string
    # uvh5_header.create_dataset("timesys", data=) # 0 string
    # uvh5_header.create_dataset("x_orientation", data=) # 0 string
    # uvh5_header.create_dataset("uvplane_reference_time", data=) # 0 int

    uvh5_header.create_dataset("phase_center_ra", data=head_dict['phase_center_ra'], dtype='d')
    uvh5_header.create_dataset("phase_center_dec", data=head_dict['phase_center_dec'], dtype='d')
    uvh5_header.create_dataset("phase_center_epoch", data=head_dict['phase_center_epoch'], dtype = 'd')
    uvh5_header.create_dataset("phase_center_frame", data=head_dict['phase_center_frame'])

    
    uvh5_header.create_dataset("Ntimes", data=head_dict['Ntimes'], dtype = 'i')
    uvh5_header.create_dataset("Nblts", data=head_dict['Nblts'], dtype = 'i')
    uvh5_header.create_dataset("ant_1_array", data=head_dict['ant_1_array'], dtype='i')
    uvh5_header.create_dataset("ant_2_array", data=head_dict["ant_2_array"], dtype='i')
    uvh5_header.create_dataset("uvw_array", data=head_dict['uvw_array'], dtype='d')
    uvh5_header.create_dataset("time_array", data=head_dict['time_array'], dtype='d')
    uvh5_header.create_dataset("integration_time", data=head_dict['integration_time'], dtype='d')

    uvh5_data.create_dataset("visdata", data=data_dict['visdata'], dtype='complex64')
    uvh5_data.create_dataset("flags", data=data_dict['flags'], dtype='?')
    uvh5_data.create_dataset("nsamples", data=data_dict['nsamples'], dtype='d')
    

