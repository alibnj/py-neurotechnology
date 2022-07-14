# Functions:

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import ccovf
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def snr(sig, snr_sig_win=[45, 150], snr_noise_win=[325, 430], s_r=1000):
	'''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
	Calculates the SNR of signal
	By default in the ranges given
	The input signal must be a 1D array
	s_r: sample rate
	'''
	snr = np.std(sig[int((s_r/1000)*snr_sig_win[0]):int((s_r/1000)*snr_sig_win[1])])/np.std(sig[int((s_r/1000)*snr_noise_win[0]):int((s_r/1000)*snr_noise_win[1])])
	return snr

def fix_polarity(signal, ref=2, t1=0, t2=1, t3=3, cor_region=[50, 150], max_shift=25, s_r=1000):
	'''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
	signal has a dimension of (channel number - max no_ch channels) X (sector number) X (length of signal)
	ref: index of the reference channel
	t1-t3: index of target channels (Their polarity will change)
	By default Ch 5 is the reference, and Ch 1-4-6 are targets
	cor_region: list in the form of [start, end] of the signal, used for xcorrelation. Default: [50, 150] ms
	max_shift: Maximum allowable shift, a shift necessary more than this value will results in change of polarity. Default: 25 ms
	s_r: Sample rate of the experiment. Default: 1000

	This function requires other functions to work properly: phase_allign,
	'''
	import numpy as np
	from scipy.ndimage.interpolation import shift

	sig = signal[:, :, int((s_r/1000)*0):int((s_r/1000)*500)]
	no_sec = sig.shape[1]
	for i in range(no_sec):
		s_ref  = sig[ref, i, :]
		s_t1   = sig[t1, i, :]
		s_t2   = sig[t2, i, :]
		s_t3   = sig[t3, i, :]
		shift1 = phase_align(s_ref, s_t1, cor_region)
		shift2 = phase_align(s_ref, s_t2, cor_region)
		shift3 = phase_align(s_ref, s_t3, cor_region)
		# If all of the signals are not aligning well, we should change the polarity of the reference signal:
		if (shift1>max_shift and shift2>max_shift and shift3>max_shift):
			s_ref = -s_ref.copy()
			shift1 = phase_align(s_ref, s_t1, cor_region)
			shift2 = phase_align(s_ref, s_t2, cor_region)
			shift3 = phase_align(s_ref, s_t3, cor_region)
			# Then we'll test again, if any of the channels are not correlating well within a reasonable threshold,
			# We change the polarity of that signal:
			if (shift1>max_shift):
				s_t1 = -s_t1.copy()
			elif (shift2>max_shift):
				s_t2 = -s_t2.copy()
			elif (shift3>max_shift):
				s_t3 = -s_t3.copy()
			# replacing the new polarities with the ones in the original signal:
			sig[:, i, :] = [s_t1, s_t2, s_ref, s_t3]
		# Else if any particular channel has poor correlation, it's polarity is changed:
		elif (shift1>max_shift):
			s_t1 = -s_t1.copy()
			sig[t1, i, :] = s_t1
		elif (shift2>max_shift):
			s_t2 = -s_t2.copy()
			sig[t2, i, :] = s_t2
		elif (shift3>max_shift):
			s_t3 = -s_t3.copy()
			sig[t3, i, :] = s_t3
	return sig


def align_signals(signal, ref=2, t1=0, t2=1, t3=3, cor_region=[50, 150], s_r=1000):
	'''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
	signal has a dimension of (channel number - max no_ch channels) X (sector number) X (length of signal)
	signal's polarities should be first fixed using fix_polarity function
	ref: index of the reference channel
	t1-t3: index of target channels (They will be shifted)
	By default Ch 5 is the reference, and Ch 1-4-6 are targets
	cor_region: list in the form of [start, end] of the signal, used for xcorrelation. Default: [50, 150] ms
	s_r: Sample rate of the experiment. Default: 1000

	returns: shifted signals and shift values
	This function requires other functions to work properly: phase_allign,
	'''
	import numpy as np
	from scipy.ndimage.interpolation import shift

	sig = signal[:, :, int((s_r/1000)*0):int((s_r/1000)*500)]
	no_sec = sig.shape[1]
	shifts = np.ndarray((no_sec, 3), dtype=float)
	for i in range(no_sec):
		s_ref  = sig[ref, i, :]
		s_t1   = sig[t1, i, :]
		s_t2   = sig[t2, i, :]
		s_t3   = sig[t3, i, :]
		shift1 = phase_align(s_ref, s_t1, cor_region)
		shift2 = phase_align(s_ref, s_t2, cor_region)
		shift3 = phase_align(s_ref, s_t3, cor_region)
		# Recording the shifts for target channels:
		shifts[i] = [shift1, shift2, shift3]
		# Replacing the shifted signals with the original ones:
		sig[t1, i, :] = shift(s_t1, shift1, mode='nearest') # Nearest sets the values out of the bound to the most recent one
		sig[t2, i, :] = shift(s_t2, shift2, mode='nearest')
		sig[t3, i, :] = shift(s_t3, shift3, mode='nearest')
	return sig, shifts


def w_ave_and_snr(signal, mode='snr_weighted', snr_sig_win=[45, 150], snr_noise_win=[325, 430], s_r=1000, weights=1):
	'''
    Author: Ali Banijamali (banijamali.s@northeastern.edu)
    
	signal has a dimension of (channel number - max no_ch channels) X (sector number) X (length of signal)
	modes:
		'snr_weighted': weighted average by snr of each sector (Default)
		'max_lr_snr_weighted': weighted average bymax of left and right eye SNR for each sector - It requires weights to be supplied
		'regular': regular average
	snr_sig_win: Signal window used for calculating SNR. Default: [45, 150]
	snr_noise_win: Noise window used for calculating SNR. Default: [325, 430]
	s_r: Sample rate. Default: 1000
	weights: used by max_lr_snr_weighted mode. Weights in this case should be suplied

	returns:
	[0] averaged signal (Dimension: no_sectors X length of signal)
	[1] SNRs of sectors for the averaged signal
	[2] SNR of unaveraged signal for all channels. Dim:
	This function requires other functions to work properly: snr
	'''
	import numpy as np

	sig = signal[:, :, int((s_r/1000)*0):int((s_r/1000)*500)]
	no_ch = sig.shape[0]
	no_sec = sig.shape[1]
	sig_len = sig.shape[2]
	ave_sig = np.ndarray((no_sec, sig_len), dtype=float)
	shifts = np.ndarray((no_sec, 3), dtype=float)
	sig_snr = np.ndarray((no_ch, no_sec), dtype=float)
	ave_sig_snr = np.zeros(no_sec, dtype=float)
	for i in range(no_sec):
		sig_snr[:, i] = np.std(sig[:, i, int((s_r/1000)*snr_sig_win[0]):int((s_r/1000)*snr_sig_win[1])], axis=1)/np.std(sig[:, i, int((s_r/1000)*snr_noise_win[0]):int((s_r/1000)*snr_noise_win[1])], axis=1)
		if mode=='snr_weighted':
			ave_sig[i] = np.average(sig[:, i], axis = 0, weights=sig_snr[:, i])
			ave_sig_snr[i] = snr(ave_sig[i], snr_sig_win=snr_sig_win, snr_noise_win=snr_noise_win)
		if mode=='regular':
			ave_sig[i] = np.average(sig[:, i])
			ave_sig_snr[i] = snr(ave_sig[i], snr_sig_win=snr_sig_win, snr_noise_win=snr_noise_win)
		if mode=='max_lr_snr_weighted':
			ave_sig[i] = np.average(sig[:, i], axis = 0, weights=weights[:, i])
			ave_sig_snr[i] = snr(ave_sig[i], snr_sig_win=snr_sig_win, snr_noise_win=snr_noise_win)

	return ave_sig, ave_sig_snr, sig_snr



def phase_align(reference, target, roi, res=100):
	'''
    Author:
    https://github.com/pearsonkyle/Signal-Alignment/blob/master/signal_alignment.py
    
	Cross-correlate data within region of interest at a precision of 1./res
	if data is cross-correlated at native resolution (i.e. res=1) this function
	can only achieve integer precision
	Args:
		reference (1d array/list): signal that won't be shifted
		target (1d array/list): signal to be shifted to reference
		roi (tuple): region of interest to compute chi-squared
		res (int): factor to increase resolution of data via linear interpolation

	Returns:
		shift (float): offset between target and reference signal
	'''
	# convert to int to avoid indexing issues
	ROI = slice(int(roi[0]), int(roi[1]), 1)

	# interpolate data onto a higher resolution grid
	x,r1 = highres(reference[ROI],kind='linear',res=res)
	x,r2 = highres(target[ROI],kind='linear',res=res)

	# subtract mean
	r1 -= r1.mean()
	r2 -= r2.mean()

	# compute cross covariance
	cc = ccovf(r1,r2,demean=False,unbiased=False)

	# determine if shift if positive/negative
	if np.argmax(cc) == 0:
		cc = ccovf(r2,r1,demean=False,unbiased=False)
		mod = -1
	else:
		mod = 1

	# often found this method to be more accurate then the way below
	return np.argmax(cc)*mod*(1./res)

	# # interpolate data onto a higher resolution grid
	# x,r1 = highres(reference[ROI],kind='linear',res=res)
	# x,r2 = highres(target[ROI],kind='linear',res=res)

	# # subtract off mean
	# r1 -= r1.mean()
	# r1 -= r2.mean()

	# # compute the phase-only correlation function
	# product = np.fft.fft(r1) * np.fft.fft(r2).conj()
	# cc = np.fft.fftshift(np.fft.ifft(product))

	# # manipulate the output from np.fft
	# l = reference[ROI].shape[0]
	# shifts = np.linspace(-0.5*l,0.5*l,l*res)

	# # plt.plot(shifts,cc,'k-'); plt.show()
	# return shifts[np.argmax(cc.real)]
