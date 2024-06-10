"""
Gut Sound Project
Analyse gut sound frequency and intensity

environment : gsp
"""

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import pathlib

from matplotlib import pyplot as pyplot

# %% Setup

SR = 44100
RNG = np.random.default_rng(seed=3152024)

CWD = pathlib.Path.cwd()

try:
    directory = CWD/"output path.txt"
    path = open(directory).read()
    OUTPUT_PATH = pathlib.Path(path)
    if not OUTPUT_PATH.exists():
        raise Exception
    else:
        print("Output will be saved at", OUTPUT_PATH)
except Exception:
   OUTPUT_PATH = CWD/"gen"
   OUTPUT_PATH.mkdir(exist_ok=True)
   print("Output will be saved at", OUTPUT_PATH)

# %% Individual Wave Component (IWC)

def iwc_gen(pm, E, b, f_iwc, SR, iwc_T):
    """
    Generates individual wave components (IWCs) with the a/m attributes.
    (+) - direct relationship
    (-) - inverse relationship

    Parameters
    ----------
    pm : float
        Pressure index.
        Determines relative loudness of gut sound.
    E : float
        Envelope index. Usually 0.9-1.1
        Determines max. amplitude (-) of gut sound.
    b : float
        Shape index. Usually 40-60.
        Determines max. amplitude (+) and rate of change of amplitude (+).
    f_iwc : float
        Resonant frequency of bowel wall. 0 - 600 Hz.
    SR : int
        Sample rate of sound.
    iwc_T : float
        Length of IWC (in seconds)

    Raises
    ------
    Exception
        Division error if divisor is <= 0.

    Returns
    -------
    iwc : np.array
        Amplitude of IWCs wrt. time

    """
    t = np.linspace(1E-5, iwc_T, int(iwc_T * SR), endpoint=False)
    # test divisor
    if (t**b > 0).all():
        iwc = pm * np.exp(-E / t) / (t ** b) * np.sin(2 * np.pi * f_iwc * t)
    else:
        raise Exception("Division error")
    return iwc


def piecewise_iwc(pm, E, b, f_iwc, SR, iwc_T, iwc_init, bs_T):
    """
    Generates multiple IWCs with differing initiation times and returns
    the sum amplitude.

    Parameters
    ----------
    pm : float
        Pressure index.
        Determines relative loudness of gut sound.
    E : float
        Envelope index. Usually 0.9-1.1
        Determines max. amplitude (-) of gut sound.
    b : float
        Shape index. Usually 40-60.
        Determines max. amplitude (+) and rate of change of amplitude (+).
    f_iwc : float
        Resonant frequency of bowel wall. 0 - 600 Hz.
    SR : int
        Sample rate of sound.
    iwc_T : float
        Length of IWC (in seconds)
    iwc_init : float
        Start time of IWC (in seconds)
    bs_T : float
        Total length of gut sound.

    Raises
    ------
    Exception
        Length of IWC doesn't match expected length.

    Returns
    -------
    total_iwc : np.array
        Amplitude of all IWCs wrt. time

    """
    total_iwc = np.zeros(int(bs_T*SR))
    end_index = iwc_init + int(SR*iwc_T)
    iwc = iwc_gen(pm, E, b, f_iwc, SR, iwc_T)
    # check_lengths
    seg_length = len(total_iwc[iwc_init:end_index])
    iwc_length = len(iwc)
    if seg_length == iwc_length:  # all ok
        total_iwc[iwc_init:end_index] = iwc
    elif seg_length < iwc_length:  # trim iwc
        print("trim", iwc_init)
        total_iwc[iwc_init:end_index] = iwc[:seg_length]
    else:
        raise Exception("f{iwc_init}")

    return total_iwc


def get_CIT_ratio(bs_sim):
    """
    Returns CIT ratio, which is defined as the CIT over the duration of the IWC.
    i.e. Duration of gut sound (non-silence) / Total duration

    Parameters
    ----------
    bs_sim : np.array
        Simulated gut sound.

    Returns
    -------
    None.

    """
    CIT = np.count_nonzero(bs_sim)
    IWC = len(bs_sim) - CIT
    if IWC == 0:
        ratio = np.inf()
    else:
        ratio = np.round(CIT/IWC,1)
    print("CIT_ratio", ratio)


# %%% Test IWC

# iwc_T = 0.04
# pm = 1
# E = 1 # decay constant
# b = 50
# f_iwc = 400

# iwc= iwc_gen(pm, E, b, f_iwc, SR, iwc_T)
# iwc = iwc/np.max(np.abs(iwc))
# timeseries  = np.arange(0,iwc_T,1/SR)

# sd.play(iwc,SR)
# p = pyplot.plot(timeseries,iwc)

# %% Generate BS


def generate_bs(bs_T, iwc_T=0.04,
                CQ=None, CQ_low=20, CQ_high=150,
                f_iwc=None, freq_low=100, freq_high=600,
                amp_spread=1,
                overall_amp=1,
                E=1, b=50,
                play=True, name=None):
    """
    Generates a simulated bowel sound with random values for:
        - Component Quantity (CQ)
        - IWC Frequency (f_iwc)

    Parameters
    ----------
    bs_T : float
        length (in s)
    iwc_T : float, optional
        length of iwc (in s). The default is 0.04.

    CQ : int, optional.
        Exact component quantity. If None, CQ is randomised. The default is None.
    CQ_low : int, optional
        Lower limit of component quantity (i.e. num IWCs/s). The default is 0.
    CQ_high : int, optional
        Upper limit of component quantity (i.e. num IWCs/s). The default is 300.

    f_iwc: float, optional
        Frequency. If None, f_iwc is randomised. The default is None.
    freq_low : float, optional
        Lower limit of frequency (Hz). The default is 100.
    freq_high : TYPE, optional
        Upper limit of frequency (Hz). The default is 600.

    amp_spread : float, optional
        Amplitude spread (i.e. std. dev). The default is 1.
    overall_amp: float, optional
        Overall amplitude. The default is 1.

    E : float, optional
        Envelope index. Usually 0.9-1.1
        Determines max. amplitude (-) of gut sound.
        The default is 1.
    b : float, optional
        Shape index. Usually 40-60.
        Determines max. amplitude (+) and rate of change of amplitude (+).
        The default is 50.

    play : boolean, optional
        If True, plays the sound. The default is True.
    name: string, optional
        Name of bowel sound.
        If not None, saves the sound with "{name}.wav". The default is None.

    Returns
    -------
    bs_sim : np.array
        Simulated gut sound.
    info : dict
        Characteristic of gut sound.

    """
    iwc_T = 0.04  # 50 ms iwc length
    if CQ is None:
        CQ = RNG.integers(low=CQ_low, high=CQ_high)  # number of IWC/s
    normCQ = int(CQ*bs_T/5)  # scale for each 5 s
    if f_iwc is None:
        f_iwc = RNG.integers(low=freq_low, high=freq_high)
    f_iwc_freq_spread = np.round(RNG.random()/3+0.01, 2) # relative to centroid
    iwc_init_lib = []
    iwc_lib = []
    min_iwc_init = 0
    max_iwc_init = (bs_T*SR)-(iwc_T*SR)

    print("\nCQ", CQ)
    print("Dur/s", np.round(bs_T,2))
    print("Centroid", f_iwc, "Hz")
    print("Spread", f_iwc_freq_spread)
    print("P spread", amp_spread)

    for ii in range(normCQ):  # generate start times; end_times
        # randomise
        pm = RNG.normal(1, amp_spread)
        f_iwc_temp = RNG.normal(f_iwc,
                                      f_iwc_freq_spread*f_iwc)
        iwc_init = min_iwc_init+RNG.random()*(max_iwc_init-min_iwc_init)
        iwc_init = int(iwc_init)
        iwc_lib.append(piecewise_iwc(pm, E, b, f_iwc_temp, SR,
                                     iwc_T, iwc_init, bs_T))
        iwc_init_lib.append(iwc_init)

    if len(iwc_lib) > 0:
        bs_sim = np.sum(iwc_lib, axis=0)
    elif len(iwc_lib) == 1:
        bs_sim = iwc_lib[0]
    else:
        # silence
        bs_sim = np.zeros(int(SR*bs_T))

    # Normalize the waveform
    magnitude = np.max(np.abs(bs_sim))
    if magnitude > 0:
        bs_sim /= magnitude
    if overall_amp != 1:
        bs_sim *= overall_amp

    get_CIT_ratio(bs_sim)

    # Play the sound
    if play:
        play_bs(name, bs_sim, SR)

    info = {"CQ": CQ, "Centroid": f_iwc, "Spread": f_iwc_freq_spread,
            "P spread": amp_spread}

    if name is not None:
        save_bs(name, bs_sim, bs_T, SR)

    return [bs_sim, info]


def play_bs(name, bs_sim, SR):
    """
    Plays the gut sound.

    Parameters
    ----------
    name: string
        Name of bowel sound.
    bs_sim : np.array
        Simulated gut sound.
    SR : int
        Sample rate of sound.

    Returns
    -------
    None.

    """
    print(f"\n\t###Playing {name} sound###")
    sd.play(bs_sim, SR)
    sd.wait()


def save_bs(name, bs_sim, SR):
    """
    Saves the gut sound.

    Parameters
    ----------
    name: string
        Name of bowel sound.
    bs_sim : np.array
        Simulated gut sound.
    SR : int
       Sample rate of sound.

    Returns
    -------
    None.

    """
    fn = OUTPUT_PATH/f"{name}.wav"
    sf.write(fn, bs_sim, SR)
    print(f"\n\t###Saved {name} sound###")

def plot_bs(name, bs_sim, SR, bs_T):
    """
    Plots the gut sound.

    Parameters
    ----------
    name: string
        Name of bowel sound.
    bs_sim : np.array
        Simulated gut sound.
    SR : int
       Sample rate of sound.
    bs_T : float
        Duration of gut sound (in s).

    Returns
    -------
    None.

    """
    timeseries = np.linspace(0, bs_T, num=len(bs_sim), endpoint=False)
    pyplot.plot(timeseries, bs_sim)
    fn = OUTPUT_PATH/f"{name}.png"
    pyplot.savefig(fn)
    pyplot.close()

def gen_bs_lib(numbs, bs_T):
    """
    Generates a library of gut sounds.

    Parameters
    ----------
    numbs : int
        Number of gut sounds to generate.
    bs_T : float
        Duration of gut sound (in s).

    Returns
    -------
    None.

    """
    infos = []
    for i in range(numbs):
        name = f"bs_{i:03}"
        bs_sim, info = generate_bs(bs_T, name=name, play=False)
        infos.append(info)

    df = pd.DataFrame(infos)
    fn = OUTPUT_PATH/"bs_sounds.xlsx"
    df.to_excel(fn)

# %% Stitching BS

def stitch_bs(bs_T, SR, parts=3,
              freq_low=100, freq_high=600,
              silence_chance=None,
              name=None, play=True, plot=True):
    """
    Stitches a few segments of gut sound/silence together.
    Gut sound(s) have a common envelope index (E, 0.9-1.1), shape index (B, 40-60),
    amplitude spread (0-2 std. dev.). Overall amplitude varies from 0.7-1.0.
    Segments have a chance to be silent if silent_chance is not None. This chance
    is reduced by 75% after a silent segment and increases after a non-silent segment.

    Parameters
    ----------
    bs_T : float
        Duration of gut sound (in s).
    SR : int
       Sample rate of sound.
    parts : int, optional
        No. of segments. The default is 3.
    freq_low : float, optional
        Lower limit of frequency (Hz). The default is 100.
    freq_high : TYPE, optional
        Upper limit of frequency (Hz). The default is 600.
    silence_chance : float, optional
        Starting silence chance. If not None, segments have a chance to be silent.
        The default is None.
    name: string
        Name of bowel sound. If None, defaults to 'stitch'.
        The default is None.
    play : boolean, optional
        If True, plays the sound. The default is True.
    plot : boolean, optional
        If True, plots the sound. The default is True.

    Returns
    -------
    bs_sim : np.array
        Simulated gut sound.

    """
    if name is None:
        name = "stitch"
    # randomise E; 0.9-1.1
    E = np.round(RNG.random() *0.2 + 0.9,2)
    # randomise b; 40 - 60
    b = np.round(RNG.random()*20 + 40,2)
    # randomise amp spread; 0-2
    amp_spread = np.round(RNG.random()*2,2)
    print(f'''\nStitched sound with {parts} parts E={E} and b={b};
          Silence chance = {silence_chance:.2f}''')
    remaining_T = bs_T
    bs_lib = []
    min_length = 0.2
    max_length = bs_T/3
    silence_mult = silence_chance

    for i in range(parts):
        if remaining_T <= min_length or i == parts-1: # below min. period or final part
            bs_part_T = remaining_T
            remaining_T = 0
        else:
            # choose transition points at random
            bs_part_T = RNG.random()*remaining_T*0.5
            # restrict length to min 0.2 and max 1/3 of total
            if bs_part_T > max_length:
                bs_part_T = max_length
            elif bs_part_T< min_length:
                bs_part_T = min_length
            remaining_T -= bs_part_T

        if bs_part_T <= 0:
            break #exit

        if silence_mult is not None and RNG.random() < silence_chance:
            # determine if silent
            bs = np.zeros(int(SR*bs_part_T))
            print("\nSilence")
            print("Dur/s", np.round(bs_part_T,2))
            silence_chance /= 4 #reduce
        else:
            f_iwc = RNG.integers(low=freq_low, high=freq_high)
            # choose frequencies
            overall_amp = RNG.random()*0.3 + 0.7 # amp from 0.7-1
            bs, info = generate_bs(bs_part_T, f_iwc=f_iwc, E=E, b=b,
                                   amp_spread=amp_spread,
                                   overall_amp = overall_amp,
                                   play=False)
            silence_chance += bs_part_T*silence_mult*2/bs_T #increase chance
        print(f"Silence chance: {silence_chance:.2f}")
        bs_lib.append(bs)

    bs_sim = np.concatenate(bs_lib)

    # Check Length of bs_sim
    difference  = SR*bs_T - len(bs_sim)
    if difference > 0:
        bs_sim = np.append(bs_sim, np.zeros(difference))
    elif difference < 0:
        bs_sim = bs_sim[:-difference]
    difference  = SR*bs_T - len(bs_sim)
    assert difference == 0, "bs_sim length not accurate"

    if play:
        play_bs(name, bs_sim, SR)

    if name is not None:
        save_bs(name, bs_sim, SR)
        if plot:
            plot_bs(name, bs_sim, SR, bs_T)

    return bs_sim

# %% Run

# import sys
# sys.stdout = open('output.txt','wt')
# for i in range(50):
#     silence_chance = RNG.random()
#     stitch_bs(30, SR, parts=20, name = f"bs{i:04}",
#               silence_chance=silence_chance,
#               play=False, plot=True)
