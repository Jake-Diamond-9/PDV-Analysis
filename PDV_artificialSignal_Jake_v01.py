import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy import signal
import math

# To-do
# Make initial velocity and voltage signal continuous and then sample it afterwards
# Check for same values somewhere in time array
# Make nicer spectrogram
# Get velocity back from voltage


# smooth differentiation from matlab file exchange
def smooth_diff(n):
    '''
    % A smoothed differentiation filter (digital differentiator).
    %
    % Such a filter has the following advantages:
    %
    % First, the filter involves both the smoothing operation and differentation operation.
    % It can be regarded as a low-pass differention filter (digital differentiator).
    % It is well known that the common differentiation operation amplifies the high-frequency noises.
    % Therefore, the smoothded differentiation filter would be valuable in experimental (noisy) data processing.
    %
    % Secondly, the filter coefficients are all convenient integers (simple units) except for an integer scaling factor,
    % as may be especially significant in some applications such as those in some single-chip microcomputers
    % or digital signal processors.
    %
    % Usage:
    % h=smooth_diff(n)
    % n: filter length (positive integer larger no less than 2)
    % h: filter coefficients (anti-symmetry)
    %
    % Examples:
    % smooth_demo
    %
    % Author:
    % Jianwen Luo <luojw@bme.tsinghua.edu.cn, luojw@ieee.org> 2004-11-02
    % Department of Biomedical Engineering, Department of Electrical Engineering
    % Tsinghua University, Beijing 100084, P. R. China
    %
    % References:
    % Usui, S.; Amidror, I.,
    % Digital Low-Pass Differentiation for Biological Signal-Processing.
    % IEEE Transactions on Biomedical Engineering 1982, 29, (10), 686-693.
    % Luo, J. W.; Bai, J.; He, P.; Ying, K.,
    % Axial strain calculation using a low-pass digital differentiator in ultrasound elastography.
    % IEEE Transactions on Ultrasonics Ferroelectrics and Frequency Control 2004, 51, (9), 1119-1127.
    '''

    if n >= 2 and math.floor(n) == math.ceil(n):
        if n % 2 == 1:  # is odd
            m = int(np.fix((n - 1) / 2))
            h = np.hstack(
                (-np.ones((1, m)), np.array(0).reshape(1, 1), np.ones((1, m)))) / m / (m + 1)
            return h
        else:  # is even
            m = int(np.fix(n / 2))
            h = np.hstack((-np.ones((1, m)), np.ones((1, m)))) / m ** 2
            return h
    else:
        raise TypeError(
            'The input parameter (n) should be a positive integer larger no less than 2.')


# user input parameters
total_time = 100
sample_rate = 0.000125
frac_zero_time = 0.20
frac_rise_time = 0.10
frac_steady_time = 0.45
frac_release_time = 0.25
steady_velocity = 100
end_release_velocity = -10

# create time arrays for the different sections of the wave
zero_time = np.arange(0, round(total_time * frac_zero_time, 15), sample_rate)
rise_time = np.arange(zero_time[-1], zero_time[-1] + round(total_time * frac_rise_time, 15), sample_rate)
steady_time = np.arange(rise_time[-1], rise_time[-1] + round(total_time * frac_steady_time, 15), sample_rate)
release_time = np.arange(steady_time[-1], steady_time[-1] + round(total_time * frac_release_time, 15), sample_rate)

# create velocity arrays for the different sections of the wave
zero_vel = 0 * zero_time
rise_slope = (steady_velocity - 0) / (rise_time[-1] - zero_time[-1])
rise_intercept = steady_velocity - rise_slope * rise_time[-1]
rise_vel = rise_slope * rise_time + rise_intercept
steady_vel = steady_velocity * np.ones(steady_time.shape)
release_slope = (steady_velocity - end_release_velocity) / (steady_time[-1] - release_time[-1])
release_intercept = steady_velocity - release_slope * steady_time[-1]
release_vel = release_slope * release_time + release_intercept

# concatenate in to a single time/velocity array
time = np.concatenate((zero_time, rise_time, steady_time, release_time))
velocity = np.concatenate((zero_vel, rise_vel, steady_vel, release_vel))

# ?????????????
# time array has two of the same values in it somewhere?
print(np.unique(np.diff(time)))
# ?????????????

# up-shifted PDV wavelengths
lam_target = 1550e-9
lam_reference = 1550.01e-9
freq_target = 1 / lam_target
freq_reference = 1 / lam_reference

# beat frequency
B = np.abs(freq_target - freq_reference + (2 / lam_target) * velocity)

# ?????????????
# fringe count
# plus integration constant?
n = cumtrapz(B, time)
# ?????????????

# ?????????????
# signal not up-shifted
# voltage_Re signal
phi = 0  # assume zero phase shift
F = 2 * np.pi * n + phi
i = 0 + 1j
voltage_Re = np.sin(F)
voltage_Cp = (1 / (2 * i)) * (np.exp(i * F) - np.exp(-i * F))
# ?????????????

# velocity from voltage signal
freq = np.diff(np.unwrap(np.angle(voltage_Cp)))
# ?????????????
# getting carrier frequency theoretically instead of from signal?
vel = (lam_target/2) * (freq + (freq_target - freq_reference))
vel = np.diff(np.unwrap(np.arcsin(voltage_Re)))
# ?????????????

'''
# Debjoy method
phas = np.unwrap(np.angle(voltage_Cp))
# samplerate/1e9 = 80 means 80 sample per ns; 5 ns stencil
stencil = sample_rate / 1e9 * 5
#phas = phas.reshape(phas.shape[0])
b = -smooth_diff(math.floor(stencil))
b = b.reshape(b.shape[1])
a = 1
phasD2 = signal.lfilter(b, a, phas) * sample_rate / \
         2 / np.pi  # 40*ns is smoother than 10*ns
# vel = (lam / 2) * ((phasD2 / 1e9) - cen / 1e9)
vel = (lam_target/2) * (phasD2 + (freq_target - freq_reference))
'''

# plot spectrogram
N = 512
noverlap_frac = 0.85
nfft_mult = 10
noverlap = np.floor(noverlap_frac * N)
nfft = N * nfft_mult
fs = 1 / (time[1] - time[0])
f, t, Zxx = signal.stft(voltage_Re, fs=fs, nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)
# power = 20 * (np.log10(np.abs(Zxx)))
power = np.abs(Zxx) ** 2

# plot velocity, voltage_Re, and spectrogram
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9), dpi=200)
ax1.plot(time, velocity, 'b-', linewidth=4)
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Input Velocity Signal')
ax2.plot(time[1:], voltage_Re, 'k-', linewidth=1)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Voltage')
plt_3 = ax3.imshow(power, extent=[t.min(), t.max(), f.min(), f.max()], aspect='auto', origin='lower')
fig1.colorbar(plt_3, ax=ax3)
ax3.set_title('Spectrogram')
ax4.plot(time[1:-1], vel, 'r-', linewidth=4)
ax4.set_title('Recovered Velocity Signal')
plt.tight_layout()
plt.show()
