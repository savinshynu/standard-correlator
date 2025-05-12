import numpy as np
from astropy import coordinates, time
import astropy.units as u

def meerkat_uvw(unix, direction, antpos):
    """
    Calculates and returns uvw in meters for a given time and pointing direction.
    direction is (ra,dec) as tuple in radians.
    antpos is Nant-by-3 array of antenna locations

    returns uvw (m) as Nant-by-3 array, relative to array center
    """

    phase_center = coordinates.SkyCoord(*direction, unit='rad', frame='icrs')

    x0, y0, z0 = [5109360.133,  2006852.586, -3238948.127] # Meerkat array center

    antpos = np.array(antpos)
    antpos = coordinates.EarthLocation(x=antpos[:,0], y=antpos[:,1], z=antpos[:,2], unit='m')

    datetime = time.Time(unix, format='unix')
    #print(datetime)

    # Meerkat array center location
    tel = coordinates.EarthLocation(x=x0, y=y0, z=z0, unit='m')

    tel_p, tel_v = tel.get_gcrs_posvel(datetime)
    antpos_gcrs = coordinates.GCRS(antpos.get_gcrs_posvel(datetime)[0],
                                   obstime = datetime, obsgeoloc = tel_p,
                                   obsgeovel = tel_v)
    tel_gcrs = coordinates.GCRS(tel_p,
                                   obstime = datetime, obsgeoloc = tel_p,
                                   obsgeovel = tel_v)

    uvw_frame = phase_center.transform_to(antpos_gcrs).skyoffset_frame()
    antpos_uvw = antpos_gcrs.transform_to(uvw_frame).cartesian
    tel_uvw = tel_gcrs.transform_to(uvw_frame).cartesian

    # Calculate difference from array center
    bl = antpos_uvw - tel_uvw
    nant = len(antpos_uvw)
    uvw = np.empty((nant,3))
    for iant in range(nant):
        uvw[iant,0] = bl[iant].y.value
        uvw[iant,1] = bl[iant].z.value
        uvw[iant,2] = bl[iant].x.value

    return uvw
