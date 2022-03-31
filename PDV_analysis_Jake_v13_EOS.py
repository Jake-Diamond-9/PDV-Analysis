# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:30:38 2021

@author: HP
"""

######################################## DEFINE FUNCTIONS #####################################################

# To Do
#
# why does our method of calculating velocity give the large negative values
# in the beginning of the velocity trace?
# is there a better way to filter out the carrier frequency? why a gaussian filter?
#
# clean up variable names, syntax, repeated computations, etc...
# make user guide/SOP
#
# incorporate Numba and UQpy?


from datetime import datetime
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.fft import ifft
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import scipy.integrate as integrate
from IPython.display import display
import traceback


##############################################################################################################
##############################################################################################################

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
            m = int(np.fix((n-1)/2))
            h = np.hstack(
                (-np.ones((1, m)), np.array(0).reshape(1, 1), np.ones((1, m))))/m/(m+1)
            return h
        else:  # is even
            m = int(np.fix(n/2))
            h = np.hstack((-np.ones((1, m)), np.ones((1, m))))/m**2
            return h
    else:
        raise TypeError(
            'The input parameter (n) should be a positive integer larger no less than 2.')


##############################################################################################################
##############################################################################################################

# parent function
def PDV_Analysis(save_data,
                 file,
                 exp_type,
                 rows_to_skip,
                 nrows,
                 N,
                 noverlap_frac,
                 nfft_mult,
                 req_time_pos,
                 window_time,
                 split,
                 expansion,
                 lam,
                 neighbors,
                 spacer_thickness,
                 expansion2,
                 expansion3,
                 wid1,
                 order1,
                 wid2,
                 order2,
                 user_bounds,
                 start_time_user,
                 freq_peak_user,
                 freq_low_user,
                 C0,
                 density):

    # start the function timer
    global start_time
    start_time = datetime.now()

    # if the user wants to use the automated domain finding procedure
    if user_bounds == 'no':
        try:
            PDV_Analysis_Auto(save_data,
                              file,
                              exp_type,
                              rows_to_skip,
                              nrows,
                              N,
                              noverlap_frac,
                              nfft_mult,
                              req_time_pos,
                              window_time,
                              split,
                              expansion,
                              lam,
                              neighbors,
                              spacer_thickness,
                              expansion2,
                              expansion3,
                              wid1,
                              order1,
                              wid2,
                              order2,
                              user_bounds,
                              start_time_user,
                              freq_peak_user,
                              freq_low_user,
                              C0,
                              density)

        # if the automated code fails, revert to user defined bounds
        except Exception:

            # close the previous plots, show what the error was
            plt.close('all')
            print('###########  WARNING  ###########')
            print('MOVED TO USER DEFINED BOUNDARIES')
            traceback.print_exc()

            # function for user defined boundaries
            PDV_Analysis_User(save_data,
                              file,
                              exp_type,
                              rows_to_skip,
                              nrows,
                              N,
                              noverlap_frac,
                              nfft_mult,
                              req_time_pos,
                              window_time,
                              split,
                              expansion,
                              lam,
                              neighbors,
                              spacer_thickness,
                              expansion2,
                              expansion3,
                              wid1,
                              order1,
                              wid2,
                              order2,
                              user_bounds,
                              start_time_user,
                              freq_peak_user,
                              freq_low_user,
                              C0,
                              density)

    # if user wants to go straight to manually input bounds without trying the automated process
    elif user_bounds == 'yes':
        PDV_Analysis_User(save_data,
                          file,
                          exp_type,
                          rows_to_skip,
                          nrows,
                          N,
                          noverlap_frac,
                          nfft_mult,
                          req_time_pos,
                          window_time,
                          split,
                          expansion,
                          lam,
                          neighbors,
                          spacer_thickness,
                          expansion2,
                          expansion3,
                          wid1,
                          order1,
                          wid2,
                          order2,
                          user_bounds,
                          start_time_user,
                          freq_peak_user,
                          freq_low_user,
                          C0,
                          density)


##############################################################################################################
##############################################################################################################

# analysis with automated bounds
def PDV_Analysis_Auto(save_data,
                      file,
                      exp_type,
                      rows_to_skip,
                      nrows,
                      N,
                      noverlap_frac,
                      nfft_mult,
                      req_time_pos,
                      window_time,
                      split,
                      expansion,
                      lam,
                      neighbors,
                      spacer_thickness,
                      expansion2,
                      expansion3,
                      wid1,
                      order1,
                      wid2,
                      order2,
                      user_bounds,
                      start_time_user,
                      freq_peak_user,
                      freq_low_user,
                      C0,
                      density):

    # read in data to data frame. skip the first number of rows (rows_to_skip), then
    # take the next nrows after that. this is faster than reading in the whole data set
    data = pd.read_csv(file, skiprows=int(rows_to_skip), nrows=int(nrows))
    data.columns = ['Time', 'Ampl']

    # pull data out of data frame and in to numpy arrays
    time = data['Time'].to_numpy()
    voltage = data['Ampl'].to_numpy()

    # for output table
    bounds = 'Automated'

    # calculate sample rate
    sample_rate = 1/(time[1] - time[0])

    # calculate the center frequency, the upshift
    npts = voltage.shape[0]
    spectra1 = fft(voltage)
    spectra2 = np.abs(spectra1/npts)
    spectra3 = spectra2[0:(npts//2 + 1)]
    spectra3[1:-1] = 2*spectra3[1:-1]
    w = (sample_rate*np.arange(0, (npts/2)))/npts  # w = frequency
    index = np.argmax(spectra3[99:])
    cen = w[index+99]

    # pre-filter the data
    freq = fftshift(np.arange(-len(time)/2, len(time)/2)
                    * sample_rate/len(time))
    filt_1 = 1-np.exp(-(freq - cen)**order1 / wid1**order1) - \
        np.exp(-(freq + cen)**order1 / wid1**order1)
    # this filter is a sixth order Gaussian notch with an 10 MHz rejection band
    # surrounding the beat frequency with strongest intensity in the spectrogram
    #filt_1 = (freq > cen+0.25e9) * (freq < cen-0.25e9)
    voltagefilt = ifft(fft(voltage) * filt_1)  # data after fixt is filtered
    #voltagefilt = np.concatenate((voltage_Re[0:fixt],voltagefilt))
    voltage = voltagefilt

    noverlap = np.floor(noverlap_frac*N)

    # number of sampling points to calculate the discrete Fourier transform
    nfft = N*nfft_mult

    # calculate the short time fourier transform
    f, t, Zxx = signal.stft(np.real(voltage), fs=sample_rate,
                            nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)

    # calculate power
    power = 20*(np.log10(np.abs(Zxx)))

    # plot spectrogram
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    plt.rcParams['font.size'] = '8'
    plt.subplot(221)
    plt.imshow(power,
               extent=[t.min()/1e-9, t.max()/1e-9, f.min()/1e9, f.max()/1e9],
               aspect='auto',
               origin='lower')
    plt.colorbar(label='Power (dB)')
    plt.set_cmap('viridis')
    plt.xlabel('Time (ns)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Imported Data and Pre-Filter')

    ###############################################################################
    # this block of code is to find the block of time where the spall event occurs.

    # make an array of the carrier frequency of length t
    carrier = cen*np.ones(len(t))

    # frequency array correspoding to the max power in the spectrogram
    freq_max_power = f[np.argmax(power, axis=0)]

    # difference between the signal and carrier freq
    difference = freq_max_power - carrier

    # time derivative of the difference
    dDdt = (np.diff(difference, n=1)/1e9)/(np.diff(t, n=1)/1e-9)

    # indices where the derivative is positive
    pos_der_idx = (dDdt > 0).nonzero()[0]

    # average time step
    avg_time_step = np.mean(np.diff(t))

    # amount of time where the difference must remain positive for the start point to
    # be considered valid.

    # number of steps in the difference array where it must remain positive, rounded up
    num_steps = math.ceil(req_time_pos/avg_time_step)

    # loop through the indices where the derivative is positive to find where the
    # spall event begins. we are looking for the point where the spall signal
    # increases above the carrier frequency and stays above it for at least 50ns.
    for idx in pos_der_idx:
        # skipping the first point in the difference array because the derivative
        # array has one less point then the difference array. check to see if all
        # points for 50ns beyond idx are positive. if they are positive, save the index
        # and break the loop. if not, continue the loop.
        if np.sum(difference[1:][idx:idx+num_steps] > 0.005e9) == num_steps:
            event_start_idx = idx + 1
            break
        else:
            continue

    # calculate the amount of time before and after the event begins and convert to
    # the number of indices in the time and voltage_Re arrays
    time_before = window_time*split
    time_after = window_time*(1-split)

    avg_time_step2 = np.mean(np.diff(time))

    idx_before = math.ceil(time_before/avg_time_step2)
    idx_after = math.ceil(time_after/avg_time_step2)

    # find the index where the time is closest to the event start time
    event_idx = np.argmin(np.abs(((time - time[0]) - t[event_start_idx])))

    # get the starting and ending indices for the cut time and voltage_Re
    start_idx = event_idx - idx_before
    end_idx = event_idx + idx_after

    cutt = time[start_idx:end_idx+1]
    cutv = voltage[start_idx:end_idx+1]
    ###############################################################################

    # calculate the short time fourier transform
    f, t, Zxx = signal.stft(np.real(cutv), fs=sample_rate,
                            nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)

    # calculate the power
    power = 20*(np.log10(np.abs(Zxx)))

    # calculate the frequency where the spall signal peaks and is lowest. for the
    # low frequency make it be non-zero
    freq_peak = np.max(f[np.argmax(power, axis=0)])
    freq_low = f[np.argmax(power, axis=0)]
    freq_low = np.min(freq_low[np.nonzero(freq_low)])

    # plot cut time and frequency range on the first plot as a rectangle
    anchor = [(cutt[0] - time[0])/1e-9, freq_low*(1-expansion)/1e9]
    width = (cutt[-1] - cutt[0])/1e-9
    height = (freq_peak*(1+expansion) - freq_low*(1-expansion))/1e9
    win = Rectangle(anchor, width, height,
                    edgecolor='r',
                    facecolor='none',
                    linewidth=0.5,
                    linestyle='-')
    ax1 = plt.gca()
    ax1.add_patch(win)

    # plot spectrogram in cut timeframe
    plt.subplot(222)
    plt.imshow(power,
               extent=[t.min()/1e-9, t.max()/1e-9, f.min()/1e9, f.max()/1e9],
               aspect='auto',
               origin='lower')
    plt.colorbar(label='Power (dB)')
    plt.set_cmap('viridis')
    plt.xlabel('Time (ns)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Cut Time')
    plt.ylim([(freq_low/1e9)*(1-expansion), (freq_peak/1e9)*(1+expansion)])
    c_min = np.min(power[(f >= freq_low*(1-expansion))
                   * (f <= freq_peak*(1+expansion))])
    c_max = np.max(power[(f >= freq_low*(1-expansion))
                   * (f <= freq_peak*(1+expansion))])
    plt.clim([c_min, c_max])

    # save the cut time and voltage_Re data in to the time and voltage_Re variables
    time = cutt
    voltage = cutv

    # Remove upshift
    # index where the spall event begins in the cut data. this corresponds to
    # 'event_idx' in the uncut data
    fixt = idx_before

    # filter the data
    freq = fftshift(
        np.arange(-len(time[fixt:])/2, len(time[fixt:])/2) * sample_rate/len(time[fixt:]))
    filt_2 = 1-np.exp(-(freq - cen)**order2 / wid2**order2) - \
        np.exp(-(freq + cen)**order2 / wid2**order2)
    # this filter is a sixth order Gaussian notch with an 10 MHz rejection band
    # surrounding the beat frequency with strongest intensity in the spectrogram
    #filt_1 = (freq > cen+0.25e9) * (freq < cen-0.25e9)
    # data after fixt is filtered
    voltagefilt = ifft(fft(voltage[fixt:]) * filt_2)
    voltagefilt = np.concatenate((voltage[0:fixt], voltagefilt))
    voltage = voltagefilt

    # isolate signal
    numpts = len(time)
    freq = fftshift(np.arange((-numpts/2), (numpts/2)) * sample_rate/numpts)
    filt = (freq > freq_low*(1-expansion)) * (freq < freq_peak*(1+expansion))
    voltagefilt = ifft(fft(voltage)*filt)

    # plot spectrogram with the signal isolated and the upshift filtered out. need
    # to take only the real part of the voltage_Re in order to prevent scipy from
    # giving a two-sided spectrogram output.
    f, t, Zxx = signal.stft(np.real(voltagefilt), fs=sample_rate,
                            nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)
    power = 20*(np.log10(np.abs(Zxx)))
    plt.subplot(223)
    plt.imshow(power,
               extent=[t.min()/1e-9, t.max()/1e-9, f.min()/1e9, f.max()/1e9],
               aspect='auto',
               origin='lower')
    plt.plot(t/1e-9, f[np.argmax(power, axis=0)]/1e9, 'k-', linewidth=2)
    plt.colorbar(label='Power (dB)')
    plt.set_cmap('viridis')
    plt.xlabel('Time (ns)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Isolate Signal and Upshift Filtered Out')
    plt.ylim([(freq_low/1e9)*(1-expansion), (freq_peak/1e9)*(1+expansion)])
    c_min = np.min(power[(f >= freq_low*(1-expansion))
                   * (f <= freq_peak*(1+expansion))])
    c_max = np.max(power[(f >= freq_low*(1-expansion))
                   * (f <= freq_peak*(1+expansion))])
    plt.clim([c_min, c_max])

    # calculate velocity history
    phas = np.unwrap(np.angle(voltagefilt), axis=0)
    # samplerate/1e9 = 80 means 80 sample per ns; 5 ns stencil
    stencil = sample_rate/1e9*5
    phas = phas.reshape(phas.shape[0])
    b = -smooth_diff(math.floor(stencil))
    b = b.reshape(b.shape[1])
    a = 1
    phasD2 = signal.lfilter(b, a, phas)*sample_rate / \
        2/np.pi    # 40*ns is smoother than 10*ns
    vel = (lam/2)*((phasD2/1e9)-cen/1e9)

    # calculations if it is a spall experiment
    if exp_type == 'spall':

        # start the time at zero
        t = (time-time[0])/1e-9

        # get the peak velocity
        peak_velocity = np.max(vel)

        # get the fist local minimum after the peak velocity to get the pullback
        # velocity. 'order' is the number of points on each side to compare to.
        rel_min_idx = signal.argrelmin(vel, order=neighbors)[0]
        extrema = np.append(rel_min_idx, np.argmax(vel))
        extrema.sort()
        pullback_idx = extrema[np.where(extrema == np.argmax(vel))[0][0] + 1]
        # NOT ACTUALLY PULLBACK VELOCITY
        max_tension_velocity = vel[pullback_idx]

        # THIS IS THE REAL PULLBACK VELOCITY!
        pullback_velocity2 = peak_velocity-max_tension_velocity

        # calculate the estimated strain rate and spall strength
        strain_rate_est = (0.5/C0)*(pullback_velocity2) / \
            (t[pullback_idx]*1e-9 - t[np.argmax(vel)]*1e-9)
        spall_strength_est = 0.5*density*C0*pullback_velocity2

        # plot the final velocity trace with the peak and pullback velocities.
        plt.subplot(224)
        plt.plot(t, vel, 'k-')
        plt.plot(t[np.argmax(vel)], peak_velocity, 'go')
        plt.plot(t[pullback_idx], max_tension_velocity, 'rs')
        plt.ylim([-30, np.max(vel)*1.05])
        plt.xlim([0, np.max(t)])
        plt.grid()
        plt.xlabel('Time (ns)')
        plt.ylabel('Velocity (m/s)')
        plt.title(file)
        plt.legend(['Free surface velocity',
                   'Velocity at Max Compression: %.f' % (
                       int(round(peak_velocity))),
                    'Velocity at Max Tension: %.f' % (int(round(max_tension_velocity)))],
                   loc='lower right',
                   fontsize=8.5)
        plt.tight_layout()

        # save velocity data if wanted
        if save_data == 'yes':
            velocityData = np.stack((t, vel), axis=1)
            np.savetxt(file[0:-4] + '--velocity' + '.csv',
                       velocityData, delimiter=',')

        # save figure if wanted
        if save_data == 'yes':
            plt.savefig(file[0:-4] + '--plots' + '.png', facecolor='w')

        plt.show()

    # calculations if it is a velocity experiment
    elif exp_type == 'velocity':

        # find where the velocity first goes positive. this is to cut out the large
        # artificial negative velocity at the beginning
        for i, v in enumerate(vel > 0):
            if v == True:
                v_pos_idx = i
                break
            else:
                continue

        # only use the data after v_pos_idx
        t = time[v_pos_idx:]
        t -= t[0]
        v = vel[v_pos_idx:]

        # save velocity data if wanted
        if save_data == 'yes':
            velocityData = np.stack((t, v), axis=1)
            np.savetxt(file[0:-4] + '--velocity' + '.csv',
                       velocityData, delimiter=',')

        # get position by trapezoidal integration of velocity
        position = (integrate.cumulative_trapezoid(v, t))/1e-6

        # spacer thickness for following spall experiments

        # generate area of velocity to average over
        pos_left = spacer_thickness*(1-expansion2)
        pos_right = spacer_thickness*(1+expansion2)
        pos_left_idx = np.argmin(np.abs(position-pos_left))
        pos_right_idx = np.argmin(np.abs(position-pos_right))

        # calculate impact velocity as an average. skip the first entry because the
        # velocity array is 1 longer than the position array due to the integration
        impact_vel = np.mean(v[1:][pos_left_idx:pos_right_idx])
        impact_vel_SD = np.std(v[1:][pos_left_idx:pos_right_idx])

        # plot velocity vs position. again skipping the first velocity entry to get
        # arrays of the same length
        plt.subplot(224)
        plt.plot(position, v[1:], 'k-')
        plt.ylim([-30, np.max(v)*1.15])
        plt.xlim([-10, np.max(position)])
        plt.grid()
        plt.xlabel('Position (\u03bcm)')
        plt.ylabel('Velocity (m/s)')
        plt.title(file)

        # plot rectange to show averaging area
        anchor = [position[pos_left_idx], v[1:][pos_left_idx]*(1-expansion3)]
        width = position[pos_right_idx] - position[pos_left_idx]
        height = v[1:][pos_right_idx] * \
            (1+expansion3) - v[1:][pos_left_idx]*(1-expansion3)
        win = Rectangle(anchor, width, height,
                        edgecolor='r',
                        facecolor='none',
                        linewidth=1.5,
                        linestyle='-')
        ax4 = plt.gca()
        ax4.add_patch(win)

        # add legend
        plt.legend(['Flyer velocity',
                    'Averaging Area\nImpact Velocity: %.f(%.f)' % (int(round(impact_vel)),
                                                                   int(round(impact_vel_SD)))],
                   loc='lower right',
                   fontsize=8.5)

        plt.tight_layout()

        # save figure if wanted
        if save_data == 'yes':
            plt.savefig(file[0:-4] + '--plots' + '.png', facecolor='w')

        plt.show()

        '''
        plt.figure(2,dpi=300)
        plt.plot(t/1e-9,v,'k')
        plt.xlabel('Time (ns)')
        plt.ylabel('Flyer Velocity (m/s)')
        
        
        plt.figure(3,dpi=300)
        plt.plot(t/1e-9,v,'k')
        plt.xlabel('Time (ns)')
        plt.ylabel('Flyer Velocity (m/s)')
        plt.xlim([1090,1170])
        plt.ylim([400,725])
        plt.grid('on')
        '''

    # end timer
    end_time = datetime.now()

    # create and output dataframe of the results for a spall test
    if exp_type == 'spall':
        run_data = {'Name': ['Date',
                             'Time',
                             'File Name',
                             'Experiment Type',
                             'Run Time',
                             'Bounds',
                             'Velocity at Max Compression',
                             'Time at Max Compression',
                             'Velocity at Max Tension',
                             'Time at Max Tension',
                             'Spall Strength (est)',
                             'Strain Rate (est)',
                             'Density',
                             'Bulk Wave Speed'],
                    'Value': [start_time.strftime('%b %d %Y'),
                              start_time.strftime('%I:%M %p'),
                              file,
                              exp_type.capitalize(),
                              (end_time - start_time),
                              bounds,
                              peak_velocity,
                              t[np.argmax(vel)],
                              max_tension_velocity,
                              t[pullback_idx],
                              spall_strength_est/1e9,
                              strain_rate_est,
                              density,
                              C0],
                    'Units': ['-',
                              '-',
                              '-',
                              '-',
                              'h:mm:ss',
                              '-',
                              'm/s',
                              'ns',
                              'm/s',
                              'ns',
                              'GPa',
                              '1/s',
                              'kg/m^3',
                              'm/s']}

        # create the dataframe
        data_to_export = pd.DataFrame(data=run_data)

        # save the data
        if save_data == 'yes':
            data_to_export.to_csv(
                file[0:-4] + '--results' + '.csv', index=False)

        # display the table in the notebook
        display(data_to_export)

    # create and output dataframe of the results for a velocity test
    elif exp_type == 'velocity':
        run_data = {'Name': ['Date',
                             'Time',
                             'File Name',
                             'Experiment Type',
                             'Run Time',
                             'Bounds',
                             'Impact Velocity',
                             'Standard Deviation'],
                    'Value': [start_time.strftime('%b %d %Y'),
                              start_time.strftime('%I:%M %p'),
                              file,
                              exp_type.capitalize(),
                              (end_time - start_time),
                              bounds,
                              impact_vel,
                              impact_vel_SD],
                    'Units': ['-',
                              '-',
                              '-',
                              '-',
                              'h:mm:ss',
                              '-',
                              'm/s',
                              'm/s']}

        # create data frame
        data_to_export = pd.DataFrame(data=run_data)

        # save data
        if save_data == 'yes':
            data_to_export.to_csv(
                file[0:-4] + '--results' + '.csv', index=False)

        # display data as table in notebook
        display(data_to_export)

    # create and output dataframe of all function inputs
    func_values = {'Variable': ['save_data',
                                'file',
                                'exp_type',
                                'rows_to_skip',
                                'nrows',
                                'N',
                                'noverlap_frac',
                                'nfft_mult',
                                'req_time_pos',
                                'window_time',
                                'split',
                                'expansion',
                                'lam',
                                'neighbors',
                                'spacer_thickness',
                                'expansion2',
                                'expansion3',
                                'wid1',
                                'order1',
                                'wid2',
                                'order2',
                                'user_bounds',
                                'start_time_user',
                                'freq_peak_user',
                                'freq_low_user',
                                'C0',
                                'density'],
                   'Value': [save_data,
                             file,
                             exp_type,
                             rows_to_skip,
                             nrows,
                             N,
                             noverlap_frac,
                             nfft_mult,
                             req_time_pos,
                             window_time,
                             split,
                             expansion,
                             lam,
                             neighbors,
                             spacer_thickness,
                             expansion2,
                             expansion3,
                             wid1,
                             order1,
                             wid2,
                             order2,
                             user_bounds,
                             start_time_user,
                             freq_peak_user,
                             freq_low_user,
                             C0,
                             density]}

    # create dataframe
    function_inputs = pd.DataFrame(data=func_values)

    # save function inputs
    if save_data == 'yes':
        function_inputs.to_csv(file[0:-4] + '--inputs' + '.csv', index=False)

    # display table of function inputs
    display(function_inputs)


##############################################################################################################
##############################################################################################################

# analysis with user defined bounds
def PDV_Analysis_User(save_data,
                      file,
                      exp_type,
                      rows_to_skip,
                      nrows,
                      N,
                      noverlap_frac,
                      nfft_mult,
                      req_time_pos,
                      window_time,
                      split,
                      expansion,
                      lam,
                      neighbors,
                      spacer_thickness,
                      expansion2,
                      expansion3,
                      wid1,
                      order1,
                      wid2,
                      order2,
                      user_bounds,
                      start_time_user,
                      freq_peak_user,
                      freq_low_user,
                      C0,
                      density):

    # read in data to data frame. skip the first number of rows (rows_to_skip), then
    # take the next nrows after that. this is faster than reading in the whole data set
    data = pd.read_csv(file, skiprows=int(rows_to_skip), nrows=int(nrows))
    data.columns = ['Time', 'Ampl']

    # pull data out of data frame and in to numpy arrays
    time = data['Time'].to_numpy()
    voltage = data['Ampl'].to_numpy()

    # for output table
    bounds = 'User'

    # calculate sample rate
    sample_rate = 1/(time[1] - time[0])

    # calculate the center frequency, the upshift
    npts = voltage.shape[0]
    spectra1 = fft(voltage)
    spectra2 = np.abs(spectra1/npts)
    spectra3 = spectra2[0:(npts//2 + 1)]
    spectra3[1:-1] = 2*spectra3[1:-1]
    w = (sample_rate*np.arange(0, (npts/2)))/npts  # w = frequency
    index = np.argmax(spectra3[99:])
    cen = w[index+99]
    print(cen)

    # pre-filter the data
    freq = fftshift(np.arange(-len(time)/2, len(time)/2)
                    * sample_rate/len(time))
    filt_1 = 1-np.exp(-(freq - cen)**order1 / wid1**order1) - \
        np.exp(-(freq + cen)**order1 / wid1**order1)
    # this filter is a sixth order Gaussian notch with an 10 MHz rejection band
    # surrounding the beat frequency with strongest intensity in the spectrogram
    #filt_1 = (freq > cen+0.25e9) * (freq < cen-0.25e9)
    voltagefilt = ifft(fft(voltage) * filt_1)  # data after fixt is filtered
    #voltagefilt = np.concatenate((voltage_Re[0:fixt],voltagefilt))
    voltage = voltagefilt

    noverlap = np.floor(noverlap_frac*N)

    # number of sampling points to calculate the discrete Fourier transform
    nfft = N*nfft_mult

    # calculate the short time fourier transform
    f, t, Zxx = signal.stft(np.real(voltage), fs=sample_rate,
                            nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)

    # calculate power
    power = 20*(np.log10(np.abs(Zxx)))

    # plot spectrogram
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    plt.rcParams['font.size'] = '8'
    plt.subplot(221)
    plt.imshow(power,
               extent=[t.min()/1e-9, t.max()/1e-9, f.min()/1e9, f.max()/1e9],
               aspect='auto',
               origin='lower')
    plt.colorbar(label='Power (dB)')
    plt.set_cmap('viridis')
    plt.xlabel('Time (ns)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Imported Data and Pre-Filter')
    plt.minorticks_on()

    # user input start time
    event_start_idx = np.argmin(np.abs(t-start_time_user))

    # calculate the amount of time before and after the event begins and convert to
    # the number of indices in the time and voltage_Re arrays
    time_before = window_time*split
    time_after = window_time*(1-split)

    avg_time_step2 = np.mean(np.diff(time))

    idx_before = math.ceil(time_before/avg_time_step2)
    idx_after = math.ceil(time_after/avg_time_step2)

    # find the index where the time is closest to the event start time
    event_idx = np.argmin(np.abs(((time - time[0]) - t[event_start_idx])))

    # get the starting and ending indices for the cut time and voltage_Re
    start_idx = event_idx - idx_before
    end_idx = event_idx + idx_after

    cutt = time[start_idx:end_idx+1]
    cutv = voltage[start_idx:end_idx+1]

    # calculate the short time fourier transform
    f, t, Zxx = signal.stft(np.real(cutv), fs=sample_rate,
                            nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)

    # calculate the power
    power = 20*(np.log10(np.abs(Zxx)))

    # user input min and max frequencies
    freq_peak = freq_peak_user
    freq_low = freq_low_user

    # plot cut time and frequency range on the first plot as a rectangle
    anchor = [(cutt[0] - time[0])/1e-9, freq_low/1e9]
    width = (cutt[-1] - cutt[0])/1e-9
    height = (freq_peak - freq_low)/1e9
    win = Rectangle(anchor, width, height,
                    edgecolor='r',
                    facecolor='none',
                    linewidth=0.5,
                    linestyle='-')
    ax1 = plt.gca()
    ax1.add_patch(win)

    # plot spectrogram in cut timeframe
    plt.subplot(222)
    plt.imshow(power,
               extent=[t.min()/1e-9, t.max()/1e-9, f.min()/1e9, f.max()/1e9],
               aspect='auto',
               origin='lower')
    plt.colorbar(label='Power (dB)')
    plt.set_cmap('viridis')
    plt.xlabel('Time (ns)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Cut Time')
    plt.ylim([(freq_low/1e9), (freq_peak/1e9)])
    c_min = np.min(power[(f >= freq_low) * (f <= freq_peak)])
    c_max = np.max(power[(f >= freq_low) * (f <= freq_peak)])
    plt.clim([c_min, c_max])

    # save the cut time and voltage_Re data in to the time and voltage_Re variables
    time = cutt
    voltage = cutv

    # Remove upshift
    # index where the spall event begins in the cut data. this corresponds to
    # 'event_idx' in the uncut data
    fixt = idx_before

    # filter the data
    freq = fftshift(
        np.arange(-len(time[fixt:])/2, len(time[fixt:])/2) * sample_rate/len(time[fixt:]))
    filt_2 = 1-np.exp(-(freq - cen)**order2 / wid2**order2) - \
        np.exp(-(freq + cen)**order2 / wid2**order2)
    # this filter is a sixth order Gaussian notch with an 10 MHz rejection band
    # surrounding the beat frequency with strongest intensity in the spectrogram
    #filt_1 = (freq > cen+0.25e9) * (freq < cen-0.25e9)
    # data after fixt is filtered
    voltagefilt = ifft(fft(voltage[fixt:]) * filt_2)
    voltagefilt = np.concatenate((voltage[0:fixt], voltagefilt))
    voltage = voltagefilt

    # isolate signal
    numpts = len(time)
    freq = fftshift(np.arange((-numpts/2), (numpts/2)) * sample_rate/numpts)
    filt = (freq > freq_low) * (freq < freq_peak)
    voltagefilt = ifft(fft(voltage)*filt)

    # plot spectrogram with the signal isolated and the upshift filtered out. need
    # to take only the real part of the voltage_Re in order to prevent scipy from
    # giving a two-sided spectrogram output.
    f, t, Zxx = signal.stft(np.real(voltagefilt), fs=sample_rate,
                            nperseg=N, noverlap=noverlap, nfft=nfft, boundary=None)
    power = 20*(np.log10(np.abs(Zxx)))
    plt.subplot(223)
    plt.imshow(power,
               extent=[t.min()/1e-9, t.max()/1e-9, f.min()/1e9, f.max()/1e9],
               aspect='auto',
               origin='lower')
    plt.plot(t/1e-9, f[np.argmax(power, axis=0)]/1e9, 'k-', linewidth=2)
    plt.colorbar(label='Power (dB)')
    plt.set_cmap('viridis')
    plt.xlabel('Time (ns)')
    plt.ylabel('Frequency (GHz)')
    plt.title('Isolate Signal and Upshift Filtered Out')
    plt.ylim([(freq_low/1e9), (freq_peak/1e9)])
    c_min = np.min(power[(f >= freq_low) * (f <= freq_peak)])
    c_max = np.max(power[(f >= freq_low) * (f <= freq_peak)])
    plt.clim([c_min, c_max])

    # calculate velocity history
    phas = np.unwrap(np.angle(voltagefilt), axis=0)
    # samplerate/1e9 = 80 means 80 sample per ns; 5 ns stencil
    stencil = sample_rate/1e9*5
    phas = phas.reshape(phas.shape[0])
    b = -smooth_diff(math.floor(stencil))
    b = b.reshape(b.shape[1])
    a = 1
    phasD2 = signal.lfilter(b, a, phas)*sample_rate / \
        2/np.pi    # 40*ns is smoother than 10*ns
    vel = (lam/2)*((phasD2/1e9)-cen/1e9)




    # save voltage data if wanted
    if save_data == 'yes':
        voltageData = np.stack((time, voltagefilt), axis=1)
        np.savetxt(file[0:-4] + '--voltage' + '.csv',
                   voltageData, delimiter=',')




    # calculations for spall experiments
    if exp_type == 'spall':

        # set beginning time to zero
        t = (time-time[0])/1e-9

        # get the peak velocity
        peak_velocity = np.max(vel)

        # get the fist local minimum after the peak velocity to get the pullback
        # velocity. 'order' is the number of points on each side to compare to.
        rel_min_idx = signal.argrelmin(vel, order=neighbors)[0]
        extrema = np.append(rel_min_idx, np.argmax(vel))
        extrema.sort()
        pullback_idx = extrema[np.where(extrema == np.argmax(vel))[0][0] + 1]
        # NOT ACTUALLY PULLBACK VELOCITY
        max_tension_velocity = vel[pullback_idx]

        # THIS IS THE REAL PULLBACK VELOCITY!
        pullback_velocity2 = peak_velocity-max_tension_velocity

        # calculate the estimated strain rate and spall strength
        strain_rate_est = (0.5/C0)*(pullback_velocity2) / \
            (t[pullback_idx]*1e-9 - t[np.argmax(vel)]*1e-9)
        spall_strength_est = 0.5*density*C0*pullback_velocity2

        # plot the final velocity trace with the peak and pullback velocities.
        plt.subplot(224)
        plt.plot(t, vel, 'k-')
        plt.plot(t[np.argmax(vel)], peak_velocity, 'go')
        plt.plot(t[pullback_idx], max_tension_velocity, 'rs')
        plt.ylim([-30, np.max(vel)*1.05])
        plt.xlim([0, np.max(t)])
        plt.grid()
        plt.xlabel('Time (ns)')
        plt.ylabel('Velocity (m/s)')
        plt.title(file)
        plt.legend(['Free surface velocity',
                   'Velocity at Max Compression: %.f' % (
                       int(round(peak_velocity))),
                    'Velocity at Max Tension: %.f' % (int(round(max_tension_velocity)))],
                   loc='lower right',
                   fontsize=8.5)
        plt.tight_layout()

        # save velocity data if wanted
        if save_data == 'yes':
            velocityData = np.stack((t, vel), axis=1)
            np.savetxt(file[0:-4] + '--velocity' + '.csv',
                       velocityData, delimiter=',')

        # save figure if wanted
        if save_data == 'yes':
            plt.savefig(file[0:-4] + '--plots' + '.png', facecolor='w')

        plt.show()

    # calculations for velocity experiment
    elif exp_type == 'velocity':

        # find where the velocity first goes positive. this is to cut out the large
        # artificial negative velocity at the beginning
        for i, v in enumerate(vel > 0):
            if v == True:
                v_pos_idx = i
                break
            else:
                continue

        # only use the data after v_pos_idx
        t = time[v_pos_idx:]
        t -= t[0]
        v = vel[v_pos_idx:]

        # save velocity data if wanted
        if save_data == 'yes':
            velocityData = np.stack((t, v), axis=1)
            np.savetxt(file[0:-4] + '--velocity' + '.csv',
                       velocityData, delimiter=',')

        # get position by trapezoidal integration of velocity
        position = (integrate.cumulative_trapezoid(v, t))/1e-6

        # generate area of velocity to average over
        pos_left = spacer_thickness*(1-expansion2)
        pos_right = spacer_thickness*(1+expansion2)
        pos_left_idx = np.argmin(np.abs(position-pos_left))
        pos_right_idx = np.argmin(np.abs(position-pos_right))

        # calculate impact velocity as an average. skip the first entry because the
        # velocity array is 1 longer than the position array due to the integration
        impact_vel = np.mean(v[1:][pos_left_idx:pos_right_idx])
        impact_vel_SD = np.std(v[1:][pos_left_idx:pos_right_idx])

        # plot velocity vs position. again skipping the first velocity entry to get
        # arrays of the same length
        plt.subplot(224)
        plt.plot(position, v[1:], 'k-')
        plt.ylim([-30, np.max(v)*1.15])
        plt.xlim([-10, np.max(position)])
        plt.grid()
        plt.xlabel('Position (\u03bcm)')
        plt.ylabel('Velocity (m/s)')
        plt.title(file)

        # plot rectange to show averaging area
        anchor = [position[pos_left_idx], v[1:][pos_left_idx]*(1-expansion3)]
        width = position[pos_right_idx] - position[pos_left_idx]
        height = v[1:][pos_right_idx] * \
            (1+expansion3) - v[1:][pos_left_idx]*(1-expansion3)
        win = Rectangle(anchor, width, height,
                        edgecolor='r',
                        facecolor='none',
                        linewidth=1.5,
                        linestyle='-')
        ax4 = plt.gca()
        ax4.add_patch(win)

        # add legend
        plt.legend(['Flyer velocity',
                    'Averaging Area\nImpact Velocity: %.f(%.f)' % (int(round(impact_vel)),
                                                                   int(round(impact_vel_SD)))],
                   loc='lower right',
                   fontsize=8.5)

        plt.tight_layout()

        # save figure if wanted
        if save_data == 'yes':
            plt.savefig(file[0:-4] + '--plots' + '.png', facecolor='w')

        plt.show()

    # end timer
    end_time = datetime.now()

    # create and output dataframe with results for spall experiment
    if exp_type == 'spall':
        run_data = {'Name': ['Date',
                             'Time',
                             'File Name',
                             'Experiment Type',
                             'Run Time',
                             'Bounds',
                             'Velocity at Max Compression',
                             'Time at Max Compression',
                             'Velocity at Max Tension',
                             'Time at Max Tension',
                             'Spall Strength (est)',
                             'Strain Rate (est)',
                             'Density',
                             'Bulk Wave Speed'],
                    'Value': [start_time.strftime('%b %d %Y'),
                              start_time.strftime('%I:%M %p'),
                              file,
                              exp_type.capitalize(),
                              (end_time - start_time),
                              bounds,
                              peak_velocity,
                              t[np.argmax(vel)],
                              max_tension_velocity,
                              t[pullback_idx],
                              spall_strength_est/1e9,
                              strain_rate_est,
                              density,
                              C0],
                    'Units': ['-',
                              '-',
                              '-',
                              '-',
                              'h:mm:ss',
                              '-',
                              'm/s',
                              'ns',
                              'm/s',
                              'ns',
                              'GPa',
                              '1/s',
                              'kg/m^3',
                              'm/s']}

        # create data frame
        data_to_export = pd.DataFrame(data=run_data)

        # save data
        if save_data == 'yes':
            data_to_export.to_csv(
                file[0:-4] + '--results' + '.csv', index=False)

        # display data as table in notebook
        display(data_to_export)

    # create and output dataframe for velocity experiment
    elif exp_type == 'velocity':
        run_data = {'Name': ['Date',
                             'Time',
                             'File Name',
                             'Experiment Type',
                             'Run Time',
                             'Bounds',
                             'Impact Velocity',
                             'Standard Deviation'],
                    'Value': [start_time.strftime('%b %d %Y'),
                              start_time.strftime('%I:%M %p'),
                              file,
                              exp_type.capitalize(),
                              (end_time - start_time),
                              bounds,
                              impact_vel,
                              impact_vel_SD],
                    'Units': ['-',
                              '-',
                              '-',
                              '-',
                              'h:mm:ss',
                              '-',
                              'm/s',
                              'm/s']}

        # create dataframe
        data_to_export = pd.DataFrame(data=run_data)

        # save data
        if save_data == 'yes':
            data_to_export.to_csv(
                file[0:-4] + '--results' + '.csv', index=False)

        # display output as table in notebook
        display(data_to_export)

    # create and output dataframe of function input values
    func_values = {'Variable': ['save_data',
                                'file',
                                'exp_type',
                                'rows_to_skip',
                                'nrows',
                                'N',
                                'noverlap_frac',
                                'nfft_mult',
                                'req_time_pos',
                                'window_time',
                                'split',
                                'expansion',
                                'lam',
                                'neighbors',
                                'spacer_thickness',
                                'expansion2',
                                'expansion3',
                                'wid1',
                                'order1',
                                'wid2',
                                'order2',
                                'user_bounds',
                                'start_time_user',
                                'freq_peak_user',
                                'freq_low_user',
                                'C0',
                                'density'],
                   'Value': [save_data,
                             file,
                             exp_type,
                             rows_to_skip,
                             nrows,
                             N,
                             noverlap_frac,
                             nfft_mult,
                             req_time_pos,
                             window_time,
                             split,
                             expansion,
                             lam,
                             neighbors,
                             spacer_thickness,
                             expansion2,
                             expansion3,
                             wid1,
                             order1,
                             wid2,
                             order2,
                             user_bounds,
                             start_time_user,
                             freq_peak_user,
                             freq_low_user,
                             C0,
                             density]}

    # create dataframe
    function_inputs = pd.DataFrame(data=func_values)

    # save data
    if save_data == 'yes':
        function_inputs.to_csv(file[0:-4] + '--inputs' + '.csv', index=False)

    # display as table in notebook
    display(function_inputs)


############################################# RUN FILE ########################################################
PDV_Analysis(save_data='yes',
             file='F2--20220319--00009.txt',
             exp_type='velocity',
             window_time=200e-9,
             rows_to_skip=4.65e6,
             nrows=200e3,
             N=512,
             noverlap_frac=0.85,
             nfft_mult=10,
             req_time_pos=50e-9,
             split=0.15,
             expansion=0.20,
             lam=1547,
             neighbors=2000,
             spacer_thickness=10,
             expansion2=0.05,
             expansion3=0.05,
             wid1=1e2,
             order1=2,
             wid2=0.5e9,
             order2=6,
             user_bounds='yes',
             start_time_user=1230e-9,
             freq_peak_user=4e9,
             freq_low_user=1.5e9,
             C0=1950,
             density=1200)

################################################# KEY #########################################################

# # adjust these every run
# save_plot: 'yes' or 'no', flag to save the plot
# file: name of file that you want to run
# exp_type: 'spall' or 'velocity', must match the data or could give an error
# window_time: length of the velocity trace in seconds, usually 200e-9 for spall and 600e-9 for velocity

# # usually shouldn't adjust these
# rows_to_skip: number of rows to skip when reading in the data, usually 3.95e6
# nrows: number of rows to import after skipping first rows_to_skip, usually 120e3
# N: length of each segment in spectrogram, usually 512
# noverlap_frac: amount of overlaped data as a fraction of N, usually 0.85
# nfft_mult: number of sampling points to calculate the discrete Fourier transform as a multiple of N, usually 10
# req_time_pos: amount of time for the signal to be above the carrier frequency to be a valid start point, usually 50e-9
# split: fraction of data to leave in before the start of the spall/velocity event, usually 0.10
# expansion: fraction of data above and below the max and min frequency to leave in, usually 0.20
# lam: wavelength of the PDV, usually 1550
# neighbors: number of points to compare to when finding the min point for the pullback velocity, usually 200
# spacer_thickness: spacer thickness in order to calculate impact velocity, usually 125
# expansion2: fraction of data around the spacer thickness to average the velocity over, usually 0.15
# expansion3: fraction of data above and below the velocity to draw the rectange on the velocity trace, usually 0.05
# wid1: width of gaussian pre-filter in Hz, usually 1e2
# order1: order of gaussian pre-filter, usually 2
# wid2: width of gaussian filter in Hz, usually 0.05e9
# order2: order of gaussian filter, usually 6
# user_bounds: 'yes' or 'no', if 'yes' it will force the program to use the hard coded bounds
# start_time_user: user input start time, only used if the automated process fails
# freq_peak_user: user input peak frequency bound, only used if the automated process fails
# freq_low_user: user input low frequency bound, only used if the automated process fails
# C0: bulk wave speed of spall target, m/s
# density: density of spall target, kg/m^3
