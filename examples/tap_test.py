# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from DRG_modal.modal import estimate_frf, PLSCE, pick_poles, LSFD
from DRG_modal.utils import load_tap_test_data


# %%
# Step 1 - load data from TDMS

TDMS_file_path = (
    r"H:\Shared drives\Snowboard FYP 25\Tap_tests_01_12_25\tail_binding_centre_10hits_yellow.tdms"
)

x, y, fs, channels = load_tap_test_data(TDMS_file_path)

# x, y are shape n_reps, n_channels, n_time
# This format is important and assumed by the worker functions below

print('Channels: ', channels)
print('Sampling frequency: ', fs)
print('Repeats: ', x.shape[0])
print('Inputs: ', x.shape[1])
print('Outputs: ', y.shape[1])
print('Time points: ', y.shape[2])

# %%
# Step 2 - Estimate FRF and crop to modal region

# Edit these to control the averaging in the welch
fft_args = {
    "nperseg": x.shape[-1] // 1,
    "noverlap": 0,
}

# Edit these to crop frequency range
f_low, f_high = -np.inf, np.inf 


# Estimate FRF
_H, _f = estimate_frf(x, y, fs, method='H3', fft_args=fft_args)

# Plot to check frequencyh range
plt.figure()
plt.semilogy(_f, np.abs(_H.reshape(-1, len(_f))).T, c='C0') 
plt.axvline(f_low, c='k')
plt.axvline(f_high, c='k')

# Crop f and H
idx = np.logical_and(_f > f_low, _f < f_high)
H,f = _H[..., idx], _f[idx]

print('Frequency lines: ', H.shape[-1])

# %%
# Step 3 - extract modal properties

# seslect number of modes (not model order)
n_modes = 50

# Get poles (a la polymax)
poles = PLSCE(H, f, fs, n_modes)

# Select poles from consistency diagram
lam, L = pick_poles(H, f, poles)

# extract modeshapes and modal properties
wns, zns, phi, Hhat= LSFD(H, f, lam, L, 'a')

modal_properties = {
    'wns':wns, # n_modes,
    'zns':zns, # n_modes,
    'phi':phi, # n_modes, outputs
} 

# export modal properties to file (example)
# np.save('file_name', **modal_properties)

# %%
# Step 4 - plots for sanity checking

# %matplotlib inline

# synthetic vs measured frf - look for decent agreement near resonances
# missed resonances -> increase model order and/or re pick poles.
# away from resonances we do not expect perfect agreement
plt.figure()
plt.semilogy(f, np.abs(H.reshape(-1, len(f))).T, c='C0') 
plt.semilogy(f, np.abs(Hhat.reshape(-1, len(f))).T, c='C1')
[plt.axvline(a) for a in wns]
plt.xlim([f.min(), f.max()])

# MAC matrix - we expect off diagonal terms to all be close to 0 for perfect modal model. May not be so good in practice. Values close to 1 indicate highly correlated modes.
MAC = np.abs(phi@phi.conj().T)**2/np.outer(np.sum(np.abs(phi)**2,1), np.sum(np.abs(phi)**2,1))
plt.figure()
sns.heatmap(MAC, cmap='Greens', annot=True, vmin=0, vmax=1)