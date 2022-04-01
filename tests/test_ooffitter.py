""" test_ooffitter.py

"""
# Package Header #
from src.spikedetection.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
import abc
import cProfile
import datetime
import importlib
import io
import os
import pathlib
import pstats
from pstats import Stats, f8, func_std_string
import time
import warnings

# Third-Party Packages #
from fooof.sim.gen import gen_aperiodic
from fooof import FOOOF, FOOOFGroup
import hdf5objects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat
from xltektools.hdf5framestructure import XLTEKStudyFrame

# Local Packages #
from src.spikedetection.artifactrejection.fooof.ooffitter import OOFFitter


# Definitions #
# Classes #
class R2Map(hdf5objects.basehdf5.BaseHDF5Map):
    default_attribute_names = {"file_type": "FileType",
                               "file_version": "FileVersion",
                               "subject_id": "subject_id",
                               "start": "start",
                               "end": "end",
                               "window_size": "window_size"}
    default_map_names = {"data": "data"}
    default_maps = {"data": hdf5objects.datasets.TimeSeriesMap()}


class R2HDF5(hdf5objects.BaseHDF5):
    FILE_TYPE = "R2OverTime"
    default_map = R2Map()


class StatsMicro(Stats):
    def print_stats(self, *amount):
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = " " * 8
        for func in self.top_level:
            print(indent, func_get_function_name(func), file=self.stream)

        print(indent, self.total_calls, "function calls", end=" ", file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end="  ", file=self.stream)
        print("in %.3f microseconds" % (self.total_tt * 1000000), file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            self.print_title()
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_title(self):
        print('          ncalls          tottime      percall      cumtime      percall', end=' ', file=self.stream)
        print('filename:lineno(function)', file=self.stream)

    def print_line(self, func):  # hack: should print percentages
        cc, nc, tt, ct, callers = self.stats[func]
        c = str(nc)
        if nc != cc:
            c = c + "/" + str(cc)
        print(c.rjust(20), end=" ", file=self.stream)
        print(str(f8(tt * 1000000)).rjust(12), end=" ", file=self.stream)
        if nc == 0:
            print(" " * 8, end=" ", file=self.stream)
        else:
            print(str(f8(tt / nc * 1000000)).rjust(12), end=" ", file=self.stream)
        print(str(f8(ct * 1000000)).rjust(12), end=" ", file=self.stream)
        if cc == 0:
            print(" " * 8, end=" ", file=self.stream)
        else:
            print(str(f8(ct / cc * 1000000)).rjust(12), end=" ", file=self.stream)
        print(func_std_string(func), file=self.stream)


# Functions #
def closest_square(n):
    n = int(n)
    i = int(np.ceil(np.sqrt(n)))
    while True:
        if (n % i) == 0:
            break
        i += 1
    assert n == (i * (n // i))
    return i, n // i


def get_lead_groups(el_label, el_type):
    assert len(el_label) == len(el_type)

    LEAD_NAME_NOID = np.array([''.join(map(lambda c: '' if c in '0123456789' else c, ll))
        for ll in el_label])
    CONTACT_IX = np.arange(len(el_label))
    LEAD_NAME = np.unique(LEAD_NAME_NOID)

    lead_group = {}
    for l_name in LEAD_NAME:
        lead_group[l_name] = \
            {'Contacts': el_label[np.flatnonzero(LEAD_NAME_NOID == l_name)],
             'IDs': CONTACT_IX[np.flatnonzero(LEAD_NAME_NOID == l_name)],
             'Type': np.unique(el_type[np.flatnonzero(LEAD_NAME_NOID == l_name)])}
        assert len(lead_group[l_name]['Type']) == 1

        lead_group[l_name]['Type'] = lead_group[l_name]['Type'][0]

    return lead_group


def make_bipolar(lead_group):
    for l_name in lead_group:
        sel_lead = lead_group[l_name]
        n_contact = len(sel_lead['IDs'])
        if 'grid' in sel_lead['Type']:
            n_row, n_col = closest_square(n_contact)
        else:
            n_row, n_col = [n_contact, 1]

        CA = np.arange(len(sel_lead['Contacts'])).reshape((n_row, n_col), order='F')

        lead_group[l_name]['Contact_Pairs_ix'] = []

        if n_row > 1:
            for bp1, bp2 in zip(CA[:-1, :].flatten(), CA[1:, :].flatten()):
                lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))

        if n_col > 1:
            for bp1, bp2 in zip(CA[:, :-1].flatten(), CA[:, 1:].flatten()):
                lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))

        """
        if (n_row > 1) & (n_col > 1):
            for bp1, bp2 in zip(CA[:-1, :-1].flatten(), CA[1:, 1:].flatten()):
                lead_group[l_name]['Contact_Pairs_ix'].append(
                        (sel_lead['IDs'][bp1],
                         sel_lead['IDs'][bp2]))
        lead_group[l_name]['Contact_Pairs_ix'] = np.array(
            lead_group[l_name]['Contact_Pairs_ix'])

        lead_group[l_name]['Contact_Pairs_ix'] = \
            lead_group[l_name]['Contact_Pairs_ix'][
                np.argsort(lead_group[l_name]['Contact_Pairs_ix'][:, 0])]
        """

    return lead_group


def make_bipolar_elecs_all(eleclabels, eleccoords):

    lead_group = get_lead_groups(eleclabels[:, 1], eleclabels[:, 2])
    lead_group = make_bipolar(lead_group)

    bp_elecs_all = {
            'IDX': [],
            'Anode': [],
            'Cathode': [],
            'Lead': [],
            'Contact': [],
            'Contact_Abbr': [],
            'Type': [],
            'x': [],
            'y': [],
            'z': []}

    for l_name in lead_group:
        for el_ix, el_iy in lead_group[l_name]['Contact_Pairs_ix']:
            bp_elecs_all['IDX'].append((el_ix, el_iy))
            bp_elecs_all['Anode'].append(el_ix)
            bp_elecs_all['Cathode'].append(el_iy)

            bp_elecs_all['Lead'].append(l_name)
            bp_elecs_all['Contact'].append('{}-{}'.format(eleclabels[el_ix, 1], eleclabels[el_iy, 1]))
            bp_elecs_all['Contact_Abbr'].append('{}-{}'.format(eleclabels[el_ix, 0], eleclabels[el_iy, 0]))
            bp_elecs_all['Type'].append(lead_group[l_name]['Type'])

            try:
                coord = (eleccoords[el_ix] + eleccoords[el_iy]) / 2
            except:
                coord = [np.nan, np.nan, np.nan]
            bp_elecs_all['x'].append(coord[0])
            bp_elecs_all['y'].append(coord[1])
            bp_elecs_all['z'].append(coord[2])

    bp_elecs_all = pd.DataFrame(bp_elecs_all)
    if np.core.numeric.dtype is None:
        importlib.reload(np.core.numeric)
    return bp_elecs_all.sort_values(by=['Anode', 'Cathode']).reset_index(drop=True)


def get_ECoG_sample(study_frame, time_start, time_end):
    natus_data = {}

    # Get the Sample Rate
    if study_frame.validate_sample_rate():
        natus_data['fs'] = 1024  #
    else:
        natus_data['fs'] = 1024

    # Get the minimum number of channels present in all recordings
    natus_data['min_valid_chan'] = min([shape[1] for shape in study_frame.get_shapes()])

    natus_data['data'] = study_frame.find_data_range(time_start, time_end, approx=True)

    return natus_data


def convert_ECoG_BP(natus_data, BP_ELECS):
    natus_data['data'] = (natus_data['data'].data[:, BP_ELECS['Anode'].values] -
                          natus_data['data'].data[:, BP_ELECS['Cathode'].values])

    return natus_data


def half_life(duration, fs_state):
    samples = duration / fs_state
    return np.exp(-(1/samples)*np.log(2))


def plot_time_stacked(sig, fs, wsize=10, color='k', labels=None, zscore=True, scale=3, ax=None):
    """
    Plot of the normalized signal in a stacked montage.
    Parameters
    ----------
    sig: np.ndarray, shape: [n_sample, n_ch]
        Time series signal.
    fs: float
        Sampling frequency of the signal (in Hz)
    wsize: float
        Window size in seconds.
    color: str
        Color of the plot lines.
    labels: array-like, len(n_ch)
        Labels corresponding to the channel names.
    scale: float, default=3.0
        Standard deviations of signal fluctuation by which the montage is
        vertically spaced for each channel.
    ax: matplotlib axis
        For updating the plot post-hoc.
    """
    plt.figure()
    sig = sig[...]
    n_s, n_ch = sig.shape
    ts = np.arange(0, n_s) / fs
    if labels is None:
        labels = ['Ch{}'.format(ix + 1) for ix in range(n_ch)]

    if ax is None:
        plt.figure(figsize=(24, 12))
        ax = plt.subplot(111)

    if zscore:
        sig_Z = (sig - np.nanmean(sig, axis=0)) / np.nanstd(sig, axis=0)
    else:
        sig_Z = sig

    offset = np.arange(n_ch) * scale

    for ch, sig_ch in enumerate(sig_Z.T):
        ax.plot(ts, sig_ch + offset[ch], color=color, alpha=0.5, linewidth=0.5)

        ax.hlines(offset[ch], ts[0], ts[-1], color='k', alpha=1.0, linewidth=0.2)

    ax.set_yticks(offset)
    ax.set_yticklabels(labels)

    ax.set_xlim([ts[0], ts[0] + wsize])
    ax.set_ylim([np.min(offset) - scale, np.max(offset) + scale])

    plt.tight_layout()

    return ax


def do_fitting(foo, freqs, spectrum, freq_range):
    foo.add_data(freqs, spectrum, freq_range)
    aperiodic_params_ = foo._robust_ap_fit(freqs, spectrum)
    ap_fit = gen_aperiodic(freqs, aperiodic_params_)
    r_val = np.corrcoef(spectrum, ap_fit)
    return r_val[0][1] ** 2


def do_fittings(foo, freqs, spectra, freq_range):
    r_sq = []
    for spectrum in spectra:
        r_sq.append(do_fitting(foo, freqs, spectrum, freq_range))

    return r_sq


# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
class ClassTest(abc.ABC):
    """Default class tests that all classes should pass."""
    class_ = None
    timeit_runs = 100000
    speed_tolerance = 200

    def test_instance_creation(self):
        pass


class TestOOFitter:

    def test_r_squared(self):
        # Setup #
        SUBJECT_ID = 'EC212'
        win_size = 10
        plot_window = True
        a_out_dir = pathlib.Path(f"/home/anthonyfong/ProjectData/EpilepsySpikeDetection/{SUBJECT_ID}/artifact")
        a_out_r_path = a_out_dir.joinpath(f"R2_{win_size}seconds.h5")

        # %%
        # Load Electrode Montage #
        IMG_PATH = pathlib.Path('/common/imaging/subjects/')
        IMAGING_PATH = IMG_PATH.joinpath(SUBJECT_ID, 'elecs', 'clinical_TDT_elecs_all.mat')
        IMAGING = loadmat(IMAGING_PATH.as_posix(), squeeze_me=True)

        # Generate Bipolar Montage
        BP_ELECS_ALL = make_bipolar_elecs_all(IMAGING['eleclabels'], IMAGING['elecmatrix'])

        # Remove common bad labels
        BAD_EL = ['EKG', 'REF', 'Reference']
        BP_ELECS_ALL = BP_ELECS_ALL.loc[~BP_ELECS_ALL['Lead'].isin(BAD_EL)]

        # %%
        # Load Study #
        STUDY_PATH = pathlib.Path('/common/xltek/subjects')
        study_frame = XLTEKStudyFrame(s_id=SUBJECT_ID, studies_path=STUDY_PATH)

        T0 = (study_frame.start_datetime + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                                               microsecond=0)
        T0 = T0 + datetime.timedelta(seconds=0)
        t0 = T0
        tF = T0 + datetime.timedelta(minutes=1)
        tD = datetime.timedelta(seconds=win_size)

        # Artifact Stuff
        oof = OOFFitter(sample_rate=1024.0, axis=0)
        oof.lower_frequency = 3
        oof.upper_frequency = 150
        fg = FOOOFGroup(peak_width_limits=[4, 8], min_peak_height=0.05, max_n_peaks=0, verbose=True)
        artifact_tracking = {
            'Timestamp': {'Start': [], 'End': [], 'Elapsed': []},  # General timestamp info for each window chunk

            "Timing": {"Spectra": [], "fooof": []},

            "spectra": {"r_squared": []},

            'channel_artifact': {'otl_state': [],  # Vector of historical state-values per channel
                                 'otl_decay': half_life(5, 0.5),
                                 # Duration that a single artifact occurrence should persist
                                 'otl_thresh': 1.5,
                                 # Threshold for determining whether a channel artifact has occurred
                                 'is_bad': []},  # Final binary designation of which channels are bad

            'population_artifact': {'otl_state': [],  # Vector of historical state-values for the entire population
                                    'otl_decay': half_life(60, 0.5),
                                    # Duration that a single artifact occurrence should persist
                                    'otl_welford': None,
                                    # Threshold for determining whether a population artifact has occurred
                                    'otl_thresh': 10,
                                    # Threshold for determining whether a channel artifact has occurred
                                    'is_bad': []}  # Final binary designation of whether a channel is bad
        }

        # Iterate over windows of the study frame
        while t0 <= (tF - tD):
            t0 += tD
            t1 = t0 + tD

            # print('\r{} -- {}'.format(t0, t1), end='')

            try:
                ### Try to grab current frame
                raw_ecog = get_ECoG_sample(study_frame, t0, t1)
                raw_ecog = convert_ECoG_BP(raw_ecog, BP_ELECS_ALL)
                if len(raw_ecog['data']) < (raw_ecog['fs'] * win_size):
                    raise Exception('Unable to grab full ECoG window.')

                # Spectra Rejection

                # OOF
                out = oof.fit_timeseries(data=raw_ecog["data"])

                # FOOF
                s_start = time.perf_counter()
                f_transform = np.fft.rfft(raw_ecog["data"], axis=0)
                spectra = np.square(np.abs(f_transform))
                s_timing = time.perf_counter() - s_start

                freqs = np.linspace(0, 1024.0 / 2, spectra.shape[0])

                f_start = time.perf_counter()

                fg.fit(freqs=freqs, power_spectra=spectra.T, freq_range=[3, 150], n_jobs=1)

                f_timing = time.perf_counter() - f_start

                r_squared_vector = np.array([c[2] for c in fg.group_results], ndmin=2)

                all = artifact_tracking["spectra"]["r_squared"]
                artifact_tracking["spectra"]["r_squared"].append(r_squared_vector)

                artifact_tracking["Timing"]["Spectra"].append(s_timing)
                artifact_tracking["Timing"]["fooof"].append(f_timing)

                if plot_window:
                    # Plot
                    plot_time_stacked(raw_ecog["data"], 1024)
                    fig1 = plt.figure()
                    ax1 = fig1.add_subplot(111)
                    ax1.plot(r_squared_vector.T, linestyle="", marker="o")
                    ax1.plot(out.r_squared, linestyle="", marker="o")
                    plt.show()


            except Exception as E:
                print(E)

            #### Update the artifact tracking dictionary
            artifact_tracking['Timestamp']['Start'].append(t0)
            artifact_tracking['Timestamp']['End'].append(t1)
            artifact_tracking['Timestamp']['Elapsed'].append((t0 - T0).total_seconds())

        all_r_squared = np.concatenate(artifact_tracking["spectra"]["r_squared"], axis=0)
        timestamps = np.array([dt.timestamp() for dt in artifact_tracking["Timestamp"]["Start"]])

        # Save Data
        with R2HDF5(file=a_out_r_path, create=True, build=True) as s_file:
            s_file["data"].require(data=all_r_squared, sample_rate=1.0 / win_size,
                                   start=artifact_tracking["Timestamp"]["Start"][0])
            R2_timeseries = s_file["data"]
            time_axis = R2_timeseries.time_axis
            time_axis[...] = timestamps

        # Load Data
        with R2HDF5(file=a_out_r_path, load=True) as l_file:
            R2_timeseries = l_file["data"]
            time_axis = R2_timeseries.time_axis
            fig1 = plt.figure(figsize=(10, 10), dpi=320)
            ax1 = fig1.add_subplot()
            ax1.imshow(R2_timeseries[...].T,
                       vmin=0,
                       vmax=1,
                       aspect="auto",
                       cmap='Reds')
            ax1.set_xlabel('Windows')
            ax1.set_ylabel('Channels')
            ax1.set_title('Channel-wise R2 Values')
            ax1.xaxis.set_ticks_position('bottom')

            plt.show()


# Main #
if __name__ == '__main__':
    pytest.main(["-v", "-s"])