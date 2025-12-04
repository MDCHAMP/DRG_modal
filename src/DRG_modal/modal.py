import numpy as np
from scipy.signal import stft
from warnings import warn
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange
from einops import rearrange


def estimate_frf(
    inputs: np.ndarray,
    outputs: np.ndarray,
    fs: float,
    method: str = "auto",
    fft_args: dict = {},
    rcond: float = 1e-15,
) -> tuple:
    """
    Estimate the Fourier response function for a general system using H1, H2, H3 or LSQ estimation.

    Args:
        inputs (numpy.ndarray): Shape (..., n_repeats, n_inputs, n_samples)
        outputs (numpy.ndarray): Shape (..., n_repeats, n_outputs, n_samples)
        fs (float): Sampling frequency
        method (str): 'auto', 'H1', 'H2', 'H3' or 'LSQ' to choose the estimation method. Note 'auto' decides heuristically which method to use based on the sizes of the input arrays.
        fft_args (dict): Args to be passed to `scipy.stft` i.e. 'nperseg', 'noverlap' etc.

    Returns:
        H (numpy.ndarray): Frequency response function of shape (n_outputs, n_inputs, n_freqs)
        freqs (numpy.ndarray): Frequency vector

    Notes:
        28/11/25 - updated to broadcast over all leading dims. - validated same as previous code on snowboard dataset.
        
    """

    # Heuristically decide which method to use if method is 'auto'
    r, n_inputs, n_time = inputs.shape[-3:]  # leading dims will be broadcast over
    _, n_outputs, _ = outputs.shape[-3:]
    if method == "auto":
        if n_inputs == 1:
            # SISO or SIMO
            if n_outputs > 100:
                # Big SIMO -> G_yy will be very large so use H1 only.
                method = "H1"
            else:
                # Small SIMO or SISO -> use H3 for best noise rejection.
                method = "H3"
        else:
            # MIMO -> conditioning on G_xx^{-1} will be bad. Use LSQ for better conditioning.
            method = "LSQ"

    # Warn user if conditioning is bad.
    if n_inputs > 1 and method in {"H2", "H3"}:
        warn(
            "H2 is extrememly ill conditioned for multiple inputs. Consider using a different estimator."
        )
    # Compute spectral densities
    f, t, X, ne = _sd_windowed(inputs, fs, fft_args)
    f, t, Y, ne = _sd_windowed(outputs, fs, fft_args)
    # Estimate the frequency response function based on the specified method
    if method == "H1":
        G_xy = _csd(X, Y, ne)  # ...PQN
        G_xx = _csd(X, X, ne)  # ...PPN
        inv_G_xx = np.linalg.pinv(np.moveaxis(G_xx, -1, -3), rcond)  # ...NPP
        H = np.einsum("...PQN,...NPP->...QPN", G_xy, inv_G_xx)
        # H = (G_xy.T @ np.linalg.pinv(G_xx.T, rcond)).T if not broadcasting then the above is equivalent to this
    elif method == "H2":
        G_yx = _csd(Y, X, ne)  # ...QPN
        G_yy = _csd(Y, Y, ne)  # ...QQN
        inv_G_yx = np.linalg.pinv(np.moveaxis(G_yx, -1, -3), rcond)  # ...NQP
        H = np.einsum("...QQN,...NPQ->...QPN", G_yy, inv_G_yx)  # ...QPN
        # H = (G_yy.T @ np.linalg.pinv(G_yx.T, rcond)).T
    elif method == "H3":
        # H3: Average of H1 and H2
        G_xx = _csd(X, X, ne)
        G_xy = _csd(X, Y, ne)
        G_yx = _csd(Y, X, ne)
        G_yy = _csd(Y, Y, ne)
        inv_G_xx = np.linalg.pinv(np.moveaxis(G_xx, -1, -3), rcond)  # ...NPP
        inv_G_yx = np.linalg.pinv(np.moveaxis(G_yx, -1, -3), rcond)  # ...NQP
        H1 = np.einsum("...PQN,...NPP->...QPN", G_xy, inv_G_xx)  # ...NQP
        H2 = np.einsum("...QQN,...NPQ->...QPN", G_yy, inv_G_yx)  # ...QPN
        # H1 = (G_xy.T @ np.linalg.pinv(G_xx.T, rcond)).T
        # H2 = (G_yy.T @ np.linalg.pinv(G_yx.T, rcond)).T
        H = (H1 + H2) / 2
    elif method == "LSQ":
        # Least squares solve, avoid directly computing the CSD matricies
        # average over n_repeats
        Yt = Y.mean(-4) # ...QNw
        Xt = X.mean(-4) # ...PNw
        inv_Xt = np.linalg.pinv(np.moveaxis(Xt, -2, -3), rcond)  # ...NPw
        H = np.einsum("...QNw,...NwP->...QPN", Yt, inv_Xt)
        # Yt = Y.mean(0).transpose(1, 0, 2)
        # Xt = X.mean(0).transpose(1, 0, 2)
        # H = (Yt @ np.linalg.pinv(Xt, rcond)).T
    else:
        raise ValueError("Invalid method. Choose 'auto', 'H1', 'H2', 'H3' or 'LSQ'.")

    return H, f


def _sd_windowed(x, fs, fft_args={}):
    """Broadcastable spectral density estimation.

    Computes the power spectral density of input x along its last axis using a windowing approach.
    Internal function not designed to be called externally.

    Args:
        x (np.ndarray): Shape (n_repeats, n_channels, n_samples)
        fs (float): Sampling frequency
        fft_args (dict): Keyword argument paseed to `scipy.signal.stft`

    Returns:
        np.ndarray: Array of frequency values
        np.ndarray: Array of frequency values
        np.ndarray: Spectral density values. Shape (n_repeats, n_channels, n_frequency_lines, n_windows)
        bool: Flag that is true if a multiple of 2 points were used in the fft
    """
    fft_args = {
        "nperseg": x.shape[-1] // 10,
        "noverlap": 0,
        "scaling": "psd",
        "padded": False,
        "boundary": "zeros",
        "detrend": "constant",
        "return_onesided": True,
    } | fft_args  # these defaults ensure consistency with the scipy csd function
    if "nfft" in fft_args:
        nfft = fft_args["nfft"]
    else:
        nfft = fft_args["nperseg"]
    return *stft(x, fs, **fft_args), nfft % 2


def _csd(X, Y, nfft_even=True):
    """Broadcastable cross-spectral density from spectral densities computed in `_sd_windowed`.

    Performs averages over n_repeats and n_windows to maintain consistency with `scipy.csd`.

    Args:
        X (np.ndarray): Shape (n_repeats, P, n_frequency_lines, n_windows)
        Y (np.ndarray): Shape (n_repeats, Q, n_frequency_lines, n_windows)

    Returns:
        np.ndarray: Shape (P, Q, n_frequency_lines)
    """
    Gxy = (
        np.einsum("...raNf,...rbNf->...abN", np.conj(X), Y) / X.shape[-3] / X.shape[-1]
    )
    # Now correct for the power scaling in the csd.
    if not nfft_even:
        Gxy[
            ..., 1:-1
        ] *= 2  # fft final point does not need doubling for non even data length
    else:
        Gxy[..., 1:] *= 2
    return Gxy


# %% 
# Polymax


def PLSCE(H, f, fs, n_modes):
    # f, fs in Hz!!!!!!!!!!!!!!!!!
    Q, P, N = H.shape # (outputs, inputs, frequencies)
    p = (2*n_modes) // P # (polynomial order)

    if P==1: # single input - do not use weighting
        weight = np.ones((Q, N)) 
    else:
        weight = H.std(1) # Q, N

    B = np.exp(1j*2*np.pi*f[:, None]/fs) ** np.arange(0, p+1) # freq x order
    X = np.einsum('QN,Np->QNp', weight, B) # Q, N, (p+1)
    Y = np.einsum('QPN,QN,Np->QNPp', -H, weight, B).reshape(Q, N, -1, order='F') # Q, N, P(p+1)
    R = np.einsum('QNi,QNj->Qij', X.conj(), X).real # Q, (p+1), (p+1) 
    S = np.einsum('QNi,QNj->Qij', X.conj(), Y).real # Q, (p+1), P(p+1)
    T = np.einsum('QNi,QNj->Qij', Y.conj(), Y).real # Q, P(p+1), P(p+1)
    M = 2*(T - S.swapaxes(1,2)@np.linalg.solve(R, S)).sum(0) # P(p+1), P(p+1)

    poles = {}
    for i in trange(1, p+1):
        A = np.linalg.solve(M[:P*i,:P*i],M[:P*i,P*i:P*(i+1)]) # Pi, P
        C = np.block([[np.zeros((P*(i-1),P)), np.eye(P*(i-1))],[A.T]]) #iP, iP
        dlam, V = np.linalg.eig(C)
        poles[i] = np.log(dlam)*fs, (V[:, :P] /  ((V[:, :P, None]).max(1)))# r, P
    
    return poles

# %%
# Stage 2 Pick poles from consistency diagram

def pick_poles(H, f, poles):
    # f, fs in Hz!!!!!!!!!!!!!!!!!
    # reformat poles
    all_lam = np.concatenate([a[0] for a in poles.values()])
    all_L = np.vstack([a[1] for a in poles.values()])
    rm_idx = np.logical_or(all_lam.real > 0,  np.isnan(all_lam))
    all_lam = all_lam[~rm_idx]
    all_L = all_L[~rm_idx]
    all_wn = np.abs(all_lam)/ 2 / np.pi
    # make plot
    matplotlib.use("TkAgg") # ensure plot opens interactively
    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    ax.semilogy(f, np.abs(H.reshape(-1, H.shape[-1])).T, c='C0') 
    ax2 = ax.twinx()
    for order, (cl, _) in poles.items():
        wns =  np.abs(cl) / 2 / np.pi
        zns = -np.real(cl)/wns
        wns_plot = wns[zns>0]
        ax2.scatter(wns_plot, np.ones_like(wns_plot)*order, 
                    c='k', alpha=0.5, s=0.5)
    ax.set_xlim([f.min(), f.max()])
    ax.set_title("Left-click = add pole | Right-click = undo | Middle-click or Enter = finish")
    #preallocate ouptuts and lines
    pole_idx = []
    vlines = [] 
    # setup interactive plot
    def add_vline(x_click):
        # snap to nearest wn in the data
        idx = np.abs(all_wn - x_click).argmin()
        pole_idx.append(idx)
        vline = ax.axvline(all_wn[idx], c='C1', ls='--', alpha=0.5)
        print(all_wn[idx], idx)
        vlines.append(vline)
        fig.canvas.draw_idle()

    def undo_vline():
        if pole_idx:
            pole_idx.pop()
            line = vlines.pop()
            line.remove()
            fig.canvas.draw_idle()

    def finish_and_close():
        fig.canvas.mpl_disconnect(btn_cid)
        fig.canvas.mpl_disconnect(key_cid)
        plt.close(fig)

    def on_click(event):
        if event.inaxes != ax2:
            return
        if event.button == 1:       # left click
            add_vline(event.xdata)
        elif event.button == 3:     # right click
            undo_vline()
        elif event.button == 2:     # middle click
            finish_and_close()

    def on_key(event):
        if event.key in ('enter', 'return'):
            finish_and_close()
        elif event.key == 'backspace':
            undo_vline()

    btn_cid = fig.canvas.mpl_connect('button_press_event', on_click)
    key_cid = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True) 
    matplotlib.use("module://matplotlib_inline.backend_inline")

    # format outputs
    pole_idx = np.array(pole_idx)[np.argsort(all_wn[pole_idx])] # sort by nf
    lam = all_lam[pole_idx] # continupus time poles (excluding conj pairs)
    L = all_L[pole_idx] # continupus time mpf (excluding conj pairs)
    lam = np.concatenate([lam, lam.conj()]) # add conj pairs back in
    L = np.concatenate([L, L.conj()])

    return lam, L

# %%
# Stage 3 - now extract modeshapes using 

def LSFD(H, f, lam, L, frf_type='a'):
    # f, fs in Hz!!!!!!!!!!!!!!!!!
    Q, P, N = H.shape
    
    ix = f>1e-9 # avoid div by zeros
    Hex = H[..., ix]
    sex = 2j*f[ix, None, None]*np.pi 

    # different frf types have different weightings
    if frf_type=='a': # accelerance
        lrw, urw = 0, +2
    elif frf_type=='v': # mobility
        lrw, urw = -1, +1
    elif frf_type=='d': # compliance
        lrw, urw = -2, 0
    
    # assemble parameter and data matricies and solve
    a = rearrange(1/(sex-np.diag(lam))@L, 'f r p -> (f p) r')
    LR = rearrange(np.eye(P)*sex**lrw, 'f p p2 -> (f p) p2') 
    UR = rearrange(np.eye(P)*sex**urw, 'f p p2 -> (f p) p2')
    b = rearrange(Hex, 'q p f -> (f p) q')
    a2 = np.block([a, LR, UR])
    x = np.linalg.pinv(a2)@b
    # extract modal properties
    phi = -x[:lam.shape[0]][lam.imag>0].real
    phi = phi / np.linalg.norm(phi, 2, axis=1)[:, None]
    wns = np.abs(lam[lam.imag>0]) / 2 / np.pi
    zns = -lam[lam.imag>0].real / np.abs(lam[lam.imag>0])
    # synthetic frf
    Hhat = H.copy()
    Hhat[..., ix] = rearrange(a2@x, '(f p) q -> q p f', p=P)
    # copy the predicted values where f>thresh, use H otherwise
    return wns, zns, phi, Hhat # in Hz


