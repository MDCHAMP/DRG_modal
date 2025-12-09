from nptdms import TdmsFile
import numpy as np


def load_tap_test_data(fname):
    channels = set()
    taps = 0
    for a in (obj := TdmsFile.read(fname)["Untitled"]).channels():
        channel, tap = a.name.split("_tap_")
        taps = max(taps, int(tap))
        if channel != "Hammer":
            channels.update([channel])
    fs = 1 / a.properties["wf_increment"]  # sampling frequency
    x, y = [], []

    for i in range(taps):
        x.append(obj[f"Hammer_tap_{i+1}"])
        y.append([obj[f"{channel}_tap_{i+1}"][:] for channel in channels])
    
    # Handle the case that not all taps are same length
    tot = min([len(a) for a in x])
    x = [a[:tot] for a in x]
    y = [[b[:tot] for b in a] for a in y]

    x = np.array(x)[:, None]
    y = np.array(y)

    return x, y, fs, channels