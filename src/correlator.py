import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# define logging formatter
formatter = logging.Formatter("%(asctime)s : %(name)s : %(levelname)s : %(message)s")

# define logging filehandler
filehandler = logging.FileHandler("correlator.log")
filehandler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# set up logger
logger.addHandler(filehandler)
logger.addHandler(console_handler)


# === EARLY ARGUMENT PARSING FOR THREADS ===
thread_parser = argparse.ArgumentParser(add_help=False)
thread_parser.add_argument("--nthreads", type=int, default=1)
# Parse known args to pick up nthreads, but leave the rest for the main parser
thread_args, remaining_argv = thread_parser.parse_known_args()

# === SET THREADING ENVIRONMENT VARIABLES EARLY ===
os.environ["NUMBA_NUM_THREADS"] = str(thread_args.nthreads)
os.environ["OMP_NUM_THREADS"] = str(thread_args.nthreads)
# Prevent oversubscription from BLAS, MKL, NumExpr, etc.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# === ONLY NOW IMPORT NUMBA AND NUMBA-RELEVANT LIBRARIES ===
import numba as nb

nb.set_num_threads(thread_args.nthreads)
logger.debug(f"Numba will use {nb.get_num_threads()} threads.")

# === IMPORT THE REST OF YOUR LIBRARIES ===
import time as tm

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from compute_uvw import meerkat_uvw
from katpoint import Antenna  # Meerkat library for reading metafile
from numba import complex64, complex128, njit, prange
from pyuvdata import UVData
from tqdm import tqdm
from uvh5_tools import create_uvh5

# (Optional, but good for OpenMP and affinity)
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

# Defining data annotation types
# annotation type metadata
type tmeta = dict[str, int | str | float]
# annotation type data
type tdata = tuple[np.ndarray]
type tvis = dict[str, np.ndarray]


class Correlator:
    def __init__(
        self, file_path: str, meta_file_path: str, backend: str = "cpu"
    ) -> None:
        self.file_path: str = file_path
        self.meta_file: str = meta_file_path
        self.backend: str = backend  # "cpu" or "gpu"
        self.header: tmeta | None = None
        self.data: tdata | None = None
        self.meta: tmeta = {}  # extracted information from the meta file
        self.load_header()  # loading header from the DADA file
        self.extract_meta()  # loading metadata from the observation metadata
        self.load_all_data()  # load visibilities, UVW coordinates, flag, sample ratio, baseline info, etc.

    def load_header(self):
        # Read header
        with open(self.file_path, "rb") as f:
            header = f.read(4096).decode("ascii")
        self.header = self.parse_header(header)

    def load_all_data(self, int_dur: float = 0.1) -> None:
        """
        Read the DADA file, average into integrations, correlate, and store results in `self.data`.
        """
        filesize: int = os.path.getsize(self.file_path)
        raw_size: int = filesize - int(self.header["HDR_SIZE"])
        if self.header["ORDER"] != "TAFTP":
            sys.exit("Unknown data order for Meerkat")
            logger.critical("Unknown data order for Meerkat")

        with open(self.file_path, "rb") as f:
            f.seek(self.header["HDR_SIZE"])

            nant: int = self.header["NANT"]
            nchan: int = self.header["NCHAN"]
            npol: int = self.header["NPOL"]
            ndim: int = self.header["NDIM"]
            inner_t: int = self.header["INNER_T"]

            dp = nant * nchan * inner_t * npol * ndim  # bytes per INNER_T block
            logger.debug(f"Size of each internal time block: {dp} bytes")

            check = raw_size % dp
            if check:
                sys.exit("Check the order of data")
                logger.critical("Order of the data is different than expected")

            outer_t = int(int_dur / (inner_t * float(self.header["TSAMP"]) * 1e-6))
            logger.debug(
                f"{outer_t} outer time samples within the integration window of {int_dur} s"
            )

            nint = raw_size // (dp * outer_t)  # number of integrated time samples
            logger.debug(f"Planned number of integrations: {nint}")

            nbls = nant * (nant + 1) // 2  # correlations and autocorrelations
            nprod = 4  # XX YY XY YX

            ant1_idx, ant2_idx = self.build_baseline_map(nant)

            logger.debug(
                "Initializing arrays for holding the voltage chunk and correlated visibilities"
            )
            vis_buf = np.empty((nbls, nchan, nprod), np.complex64)
            uvw_buf = np.empty((nbls, 3), np.float32)
            ant1_buf = np.empty(nbls, np.int32)
            ant2_buf = np.empty_like(ant1_buf)

            vis_mat = np.empty((nint, nbls, nchan, nprod), np.complex64)
            uvw_array = np.empty((nint, nbls, 3), np.float32)
            ant1_array = np.empty((nint, nbls), np.int32)
            ant2_array = np.empty_like(ant1_array)
            flag_mat = np.zeros_like(vis_mat, dtype=bool)
            nsamples_mat = np.ones_like(vis_mat, dtype=np.float32)

            data_offset = int(self.header["OBS_OFFSET"])
            time_offset = (data_offset // dp) * (
                inner_t * float(self.header["TSAMP"]) * 1e-6
            )

            ant_names_str: list[str] = list(self.meta["antenna_positions"].keys())
            ant_names: list[int] = [int(a[1:]) for a in ant_names_str]

            time_array = (
                float(self.header["UTC_START"])
                + time_offset
                + int_dur / 2.0
                + np.arange(nint) * int_dur
            )

            antpos = np.array(list(self.meta["antenna_positions"].values()))
            pointing = self.convert_dir2float(self.header["RA"], self.header["DEC"])

            self.meta.update(
                {
                    "nTimesteps": nint,
                    "ant_index": ant_names,
                    "ant_names_str": ant_names_str,
                    "time_array": time_array,
                    "pointing": pointing,
                    "tInt": int_dur,
                }
            )

            logger.info(
                f"Reading DADA and correlating ({self.backend.upper()} backend)…"
            )
            logger.info(f"Processing {nint} data chunks")

            tf0 = tm.time()
            for num in tqdm(range(nint), disable=False):
                t0 = tm.time()

                chunk = np.fromfile(f, dtype=np.int8, count=dp * outer_t)
                if chunk.size < dp * outer_t:
                    samp_ratio = round(chunk.size / (dp * outer_t), 3)
                    nsamples_mat[num] = samp_ratio

                chunk = (
                    chunk.reshape(outer_t, nant, nchan, inner_t, npol, ndim)
                    .transpose(1, 2, 0, 3, 4, 5)
                    .reshape(nant, nchan, outer_t * inner_t, npol, ndim)
                    .astype(np.float32)
                    .view("complex64")
                    .squeeze()
                )

                uvw_now: np.ndarray = meerkat_uvw(
                    time_array[num], pointing, antpos
                ).astype(np.float32)

                # Call correct correlator based on backend
                if self.backend == "gpu":
                    # Move arrays to JAX device
                    logger.debug(f"Moving chunk:{num} to GPU")

                    chunk_jax = jnp.asarray(chunk)
                    uvw_now_jax = jnp.asarray(uvw_now)
                    ant1_idx_jax = jnp.asarray(ant1_idx)
                    ant2_idx_jax = jnp.asarray(ant2_idx)

                    vis_jax, uvw_jax, ant1_jax, ant2_jax = (
                        Correlator.calc_vis_uvw_ant_gpu(
                            chunk_jax, uvw_now_jax, ant1_idx_jax, ant2_idx_jax
                        )
                    )
                    vis_buf[:] = np.array(vis_jax)
                    uvw_buf[:] = np.array(uvw_jax)
                    ant1_buf[:] = np.array(ant1_jax)
                    ant2_buf[:] = np.array(ant2_jax)
                else:
                    Correlator.calc_vis_uvw_ant_cpu(
                        chunk,
                        uvw_now,
                        ant1_idx,
                        ant2_idx,
                        vis_buf,
                        uvw_buf,
                        ant1_buf,
                        ant2_buf,
                    )

                vis_mat[num] = vis_buf
                uvw_array[num] = uvw_buf
                ant1_array[num] = ant1_buf
                ant2_array[num] = ant2_buf

                logger.debug(
                    f"Chunk: {num}: Loading + correlation: {tm.time() - t0:0.3f}s"
                )

            logger.debug(f"Total elapsed inside loop: {tm.time() - tf0:0.3f}s")

        ant_numbers_lookup = np.asarray(self.meta["ant_index"], dtype=np.int32)
        ant1_array = ant_numbers_lookup[ant1_array]
        ant2_array = ant_numbers_lookup[ant2_array]

        nprod = 4
        self.data = (
            vis_mat.reshape(nint * nbls, nchan, nprod),
            uvw_array.reshape(nint * nbls, 3),
            ant1_array.reshape(nint * nbls),
            ant2_array.reshape(nint * nbls),
            flag_mat.reshape(nint * nbls, nchan, nprod),
            nsamples_mat.reshape(nint * nbls, nchan, nprod),
        )

    @staticmethod
    def build_baseline_map(nant: int) -> tuple[np.ndarray]:
        """
        Create a baseline index map given the antenna numbers.
        """
        ant1 = []
        ant2 = []
        for a in range(nant):
            ant1.append(a)
            ant2.append(a)
        for a1 in range(nant):
            for a2 in range(a1 + 1, nant):
                ant1.append(a1)
                ant2.append(a2)
        return np.array(ant1, dtype=np.int32), np.array(ant2, dtype=np.int32)

    @staticmethod
    @jax.jit
    def calc_vis_uvw_ant_gpu(
        chunk: np.ndarray,
        uvw_now: np.ndarray,
        ant1_idx: np.ndarray,
        ant2_idx: np.ndarray,
    ) -> tuple[np.ndarray]:
        """
        Correlation computation in GPUs

        Returns: Visibiilities
        """
        nbls = ant1_idx.shape[0]
        nchan = chunk.shape[1]
        nt = chunk.shape[2]

        def correlate_bl(bl):
            a1 = ant1_idx[bl]
            a2 = ant2_idx[bl]

            def correlate_chan(f):
                x1 = chunk[a1, f, :, 0].astype(jnp.complex64)
                x2 = chunk[a2, f, :, 0].astype(jnp.complex64)
                y1 = chunk[a1, f, :, 1].astype(jnp.complex64)
                y2 = chunk[a2, f, :, 1].astype(jnp.complex64)
                s_xx = jnp.sum(x1 * jnp.conj(x2))
                s_yy = jnp.sum(y1 * jnp.conj(y2))
                s_xy = jnp.sum(x1 * jnp.conj(y2))
                s_yx = jnp.sum(y1 * jnp.conj(x2))
                norm = 1.0 / nt
                return jnp.stack([s_xx, s_yy, s_xy, s_yx]) * norm

            vis_bl = jax.vmap(correlate_chan)(jnp.arange(nchan))  # (nchan, 4)
            uvw = uvw_now[a1] - uvw_now[a2]
            return vis_bl, uvw, a1, a2

        vis_all, uvw_all, ant1_all, ant2_all = jax.vmap(correlate_bl)(jnp.arange(nbls))
        return vis_all, uvw_all, ant1_all, ant2_all

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def calc_vis_uvw_ant_cpu(
        chunk: np.ndarray,
        uvw_now: np.ndarray,  # inputs
        ant1_idx: np.ndarray,
        ant2_idx: np.ndarray,  # baseline map
        vis_out: np.ndarray,
        uvw_out: np.ndarray,  # outputs (pre‑allocated)
        ant1_out: np.ndarray,
        ant2_out: np.ndarray,
    ):
        """
        Correlation computation in CPUs
        Returns visibilities
        """
        nbls, nchan = vis_out.shape[0], vis_out.shape[1]
        nt = chunk.shape[2]

        for bl in prange(nbls):
            a1 = ant1_idx[bl]
            a2 = ant2_idx[bl]

            for f in range(nchan):
                s_xx = complex128(0.0 + 0.0j)
                s_yy = complex128(0.0 + 0.0j)
                s_xy = complex128(0.0 + 0.0j)
                s_yx = complex128(0.0 + 0.0j)

                for t in range(nt):
                    x1 = complex128(chunk[a1, f, t, 0])
                    x2 = complex128(chunk[a2, f, t, 0].conjugate())
                    y1 = complex128(chunk[a1, f, t, 1])
                    y2 = complex128(chunk[a2, f, t, 1].conjugate())

                    s_xx += x1 * x2
                    s_yy += y1 * y2
                    s_xy += x1 * y2
                    s_yx += y1 * x2

                norm = 1.0 / nt
                vis_out[bl, f, 0] = complex64(s_xx * norm)
                vis_out[bl, f, 1] = complex64(s_yy * norm)
                vis_out[bl, f, 2] = complex64(s_xy * norm)
                vis_out[bl, f, 3] = complex64(s_yx * norm)

            ant1_out[bl] = a1
            ant2_out[bl] = a2
            uvw_out[bl, 0] = uvw_now[a1, 0] - uvw_now[a2, 0]
            uvw_out[bl, 1] = uvw_now[a1, 1] - uvw_now[a2, 1]
            uvw_out[bl, 2] = uvw_now[a1, 2] - uvw_now[a2, 2]

    @staticmethod
    def parse_header(header: str) -> tmeta:
        """
        Parsing the data header information into a dictionary.
        """
        header_dict = {}
        for line in header.split("\n"):
            if line and not line.startswith("#") and " " in line:
                key, value = line.split(None, 1)
                header_dict[key] = value.strip()
        header_dict["NBIT"] = int(header_dict["NBIT"])
        header_dict["NDIM"] = int(header_dict["NDIM"])
        header_dict["NPOL"] = int(header_dict["NPOL"])
        header_dict["NCHAN"] = int(header_dict["NCHAN"])
        header_dict["NANT"] = int(header_dict["NANT"])
        header_dict["INNER_T"] = int(header_dict["INNER_T"])
        header_dict["HDR_SIZE"] = int(header_dict["HDR_SIZE"])
        header_dict["ORDER"] = str(header_dict["ORDER"])
        header_dict["CHAN0_IDX"] = int(header_dict["CHAN0_IDX"])
        header_dict["CHAN_WIDTH"] = float(header_dict["OBS_BW"]) / float(
            header_dict["OBS_NCHAN"]
        )
        header_dict["FBEG"] = float(header_dict["FREQ"])
        return header_dict

    @staticmethod
    def ant2bls(ant1: int, ant2: int) -> int:
        """
        Convert antenna index to baseline indec
        """

        (a1, a2) = sorted((ant1, ant2))
        # does not work when ant1 == ant2
        return (a2 * (a2 - 1)) // 2 + a1

    @staticmethod
    def convert_dir2float(ra: float, dec: float) -> tuple[float]:
        """
        Convert the ra and dec string into
        radians
        """
        c = SkyCoord(
            ra + " " + dec, unit=(u.hourangle, u.deg)
        )  # Calling astropy to convert ra, dec string to radians
        return (c.ra.radian, c.dec.radian)  # returning ra, dec tuple in radians

    def extract_meta(self):
        """
        Extract all important information from the metafile
        """
        ant_pos = {}  # ECEF coordinates of each antenna elements
        with h5py.File(self.meta_file) as hf:
            # Parse out antenna to F-engine mapping, {needed to understand the actively data collecting antennas}
            antenna_feng_map = {
                antenna.decode(): index for antenna, index in hf["antenna_feng_map"][()]
            }

            # parse out antenna information
            for ant_info in hf["antenna_positions"]:
                # get an instance of Antenna class
                ant_ob = Antenna(ant_info.decode())
                if ant_ob.name in antenna_feng_map.keys():
                    ant_pos[ant_ob.name] = (
                        ant_ob.position_ecef
                    )  # assign the corresponding ECEF coordinates in tuples

            self.meta["antenna_positions"] = ant_pos
            self.meta["antenna_feng_map"] = antenna_feng_map
            self.meta.update(dict(hf.attrs))

    def get_header_data(self) -> tuple[tmeta, tvis]:
        """
        Collect all the data and metadata for the
        UVH5 file
        """
        lat_mkat = (
            -30.711055553291935
        )  # latitude degrees # obtained from Meerkat visibilities
        lon_mkat = 21.443888889697842  # longitude degrees
        alt_mkat = 1086.599484886974  # altitude in meters
        tel_name = self.header["TELESCOPE"]  # telescope name
        instrument = self.header["INSTRUMENT"]  # instrument name
        history = "Ask savin"
        nants_data = self.header["NANT"]  # antennas present in the data
        nants_tel = 64  # antennas present in the telescope
        nbls = int(
            nants_data * (nants_data + 1) / 2.0
        )  # number of baselines, included autocorrelations as well
        ntimes = self.meta["nTimesteps"]  # time samples
        nbltimes = nbls * ntimes  # baseline * ntimes
        nfreqs = self.header["NCHAN"]  # No. of frequency channels
        nspws = 1  # spectral windows
        npols = 4  # polarization products
        ## collecting important data
        vis_data, uvw_array, ant1_array, ant2_array, flag_data, nsamples_data = (
            self.data
        )
        ant_names = [name.encode() for name in self.meta["ant_names_str"]]
        ant_numbers = self.meta["ant_index"]
        freq_array = (
            self.header["FBEG"]
            + np.arange(self.header["NCHAN"]) * self.header["CHAN_WIDTH"]
        )  # frequency array in Hz
        chan_width = self.header["CHAN_WIDTH"] * np.ones(
            self.header["NCHAN"]
        )  # channel width in Hz
        antenna_diameter = (
            np.ones(self.header["NANT"]) * 13.5
        )  # antenna diameter of each dish, for meetkat = 13.5 m
        antenna_positions = (
            self.get_antenna_positions_ref()
        )  # ECEF coordinates relative to the reference position of the array
        integration_time = np.ones(nbltimes) * self.meta["tInt"]
        # time_array_unix = self.stamp_meta['tstart'] + np.arange(ntimes)*self.stamp_meta['tsamp'] # creating time array for the stamp in UNIX format
        time_array_unix_astro = time.Time(
            self.meta["time_array"], format="unix"
        )  # loading unix time stamps to astropy
        time_array_jd = (
            time_array_unix_astro.jd
        )  # coverting time stamps to JD day format for UVH5
        time_bls_array = np.repeat(
            time_array_jd, nbls
        )  # converting that to ntimes * nbaselines
        spw_array = np.ones((1), dtype="int32")  # only one spectral index
        flex_spw = False  # set to true if more than 1 spectral windows
        # pol_array = np.array([-1, -3, -4, -2], dtype='int32') # RR, RL, LR, LL [-1, -3, -4, -2]
        pol_array = np.array(
            [-5, -6, -7, -8]
        )  # XX, YY, XY, YX [-5, -6, -7, -8], currently only have XX and YY
        version = "1.0".encode()
        object = self.header["SOURCE"].encode()
        phase_type = "phased".encode()  # assuming the input data is phased
        phase_center_ra = self.meta["pointing"][0]  # ra and dec n radians
        phase_center_dec = self.meta["pointing"][1]
        phase_center_epoch = 2000.0  # assuming coordinates are in J2000 epoch
        phase_center_frame = "icrs".encode()
        extra_keywords = ""
        # Other data
        # flag_data = np.zeros(visdata.shape, dtype = 'bool')
        # nsamples = np.ones(visdata.shape, dtype = 'float32')

        head_dict = {
            "Nants_data": nants_data,
            "Nants_telescope": nants_data,
            "Nbls": nbls,
            "Nblts": nbltimes,
            "Nfreqs": nfreqs,
            "Npols": npols,
            "Nspws": nspws,
            "Ntimes": ntimes,
            "altitude": alt_mkat,
            "ant_1_array": ant1_array,
            "ant_2_array": ant2_array,
            "antenna_diameters": antenna_diameter,
            "antenna_names": ant_names,
            "antenna_numbers": ant_numbers,
            "antenna_positions": antenna_positions,
            "channel_width": chan_width,
            "extra_keywords": extra_keywords,
            "flex_spw": flex_spw,
            "freq_array": freq_array,
            "history": history,
            "instrument": instrument,
            "integration_time": integration_time,
            "latitude": lat_mkat,
            "longitude": lon_mkat,
            "object_name": object,
            "phase_center_dec": phase_center_dec,
            "phase_center_epoch": phase_center_epoch,
            "phase_center_frame": phase_center_frame,
            "phase_center_ra": phase_center_ra,
            "phase_type": phase_type,
            "polarization_array": pol_array,
            "spw_array": spw_array,
            "telescope_name": tel_name,
            "time_array": time_bls_array,
            "uvw_array": uvw_array,
            "version": version,
        }

        data_dict = {"flags": flag_data, "nsamples": nsamples_data, "visdata": vis_data}
        # print(head_dict)
        return head_dict, data_dict

    def get_antenna_positions_ref(self, ref_ant: int | None = None) -> np.ndarray:
        """
        Collect the antenna positions wrt to the reference antenna in the ECEF format
        If no reference antenna given, use the array center location
        """
        ant_pos = self.meta["antenna_positions"]  # dictionary containing values

        if ref_ant:
            ref_ecef = ant_pos[ref_ant]
            logger.debug(
                "Using the input reference antenna location for UVW calculation"
            )
        else:
            # Use the the coordinates of the center of the array
            ref_ecef = (5109360.133, 2006852.586, -3238948.127)
            logger.debug("Using the array center location for UVW calculation")

        ant_pos_ecef = np.array(
            list(self.meta["antenna_positions"].values())
        )  # Actual X, Y, Z antenna positions in ECEF (m)
        return ant_pos_ecef - np.array(
            ref_ecef
        )  # Antenna positions in XYZ wrt to reference antenna or center of the array

    def write_uvh5(self, outpath: str, msdata: bool, rem_uvh5: bool):
        """
        Write the header and data into a uvh5 file
        """
        os.makedirs(outpath, exist_ok=True)
        logger.debug(f"Created directory {outpath} if not exist already")

        filepath_uvh5 = os.path.join(
            outpath, os.path.splitext(os.path.basename(self.file_path))[0] + ".uvh5"
        )
        logger.info(f"Writing out {filepath_uvh5}")
        fob = h5py.File(filepath_uvh5, "w")  # creating the uvh5 file
        head_dict, data_dict = (
            self.get_header_data()
        )  # collecting all the important data and header
        create_uvh5(
            fob, head_dict, data_dict
        )  # Writing all the data into the uvh5 file handle
        fob.close()  # close afer after writing

        if msdata:  # if needed to convert the UVH5 data into the CASA MS format
            logger.info("Writing out the CASA MS format file")
            uvd = UVData()
            uvd.read(filepath_uvh5, fix_old_proj=False)
            outfile_ms = os.path.join(
                outpath, os.path.splitext(os.path.basename(self.file_path))[0] + ".ms"
            )
            if not os.path.exists(outfile_ms):
                uvd.write_ms(outfile_ms)
            else:
                logger.debug(f"{outfile_ms} already exists")
                pass

            if rem_uvh5 and os.path.exists(
                filepath_uvh5
            ):  # Remove the UVH5 file after creation of the MS file
                logger.debug(f"Removing {filepath_uvh5}")
                os.remove(filepath_uvh5)


def main(args):
    t0 = tm.perf_counter()
    fob = Correlator(args.DADAfile, args.METAfile, backend=args.backend)
    # print(fob.header)
    fob.write_uvh5(outpath=args.outdir, msdata=args.casa_ms, rem_uvh5=args.rem_uvh5)
    t1 = tm.perf_counter()
    logger.info(f"Total elapsed time: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read DADA files, crosscorrelate and write out the visibilities in the UVH5/CASA MS formats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[thread_parser],
    )
    parser.add_argument(
        "DADAfile", type=str, help="Input voltage data file in the .dada format"
    )
    parser.add_argument(
        "METAfile",
        type=str,
        help="Input metafile for the observations in the .hdf5 format",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        required=False,
        default=".",
        help="Output directory for visibilities",
    )
    parser.add_argument(
        "-ms",
        "--casa_ms",
        action="store_true",
        help="Output visibilities in CASA MS format",
    )
    parser.add_argument(
        "-r",
        "--rem_uvh5",
        action="store_true",
        help="Remove the UVH5 file after the conversion to CASA MS format",
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Select backend: cpu or gpu",
    )
    args = parser.parse_args(remaining_argv)
    main(args)
