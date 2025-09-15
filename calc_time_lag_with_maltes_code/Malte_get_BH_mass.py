#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:47:15 2024

@author: Malte
"""
import numpy as np

class Line():
    def __init__(self, name, FWHM_rms, FWHM_rms_errs, correlation, tau_cents, tau_peaks, which='cent', top_percent=0.8, **kwargs):
        self.name = name
        self.FWHM_rms = FWHM_rms
        self.FWHM_rms_errs = FWHM_rms_errs
        self.correlation = correlation
        self.tau_cents = tau_cents
        self.tau_peaks = tau_peaks
        self.tau_cent, self.tau_cent_err = self.getTauCent(self.correlation, self.tau_cents, top_percent=top_percent)
        self.R_cent, self.R_cent_err = self.resizeSI(self.tau_cent, self.tau_cent_err)
        self.r_max, self.tau_peak, self.tau_peak_err = self.getTauPeak(self.correlation, self.tau_peaks)
        self.R_peak, self.R_peak_err = self.resizeSI(self.tau_peak, self.tau_peak_err)
        self.calcBHMass(which=which)

        
    
    def getTauCent(self, correlation, tau_cents, top_percent, **kwargs):
        threshold = top_percent * np.amax(correlation[:,1])
        mask = correlation[:,1] >= threshold
        tau_cent = np.sum(correlation[mask,1] * correlation[mask,0]) / np.sum(correlation[mask,1])
        sortedCentroids = np.sort(tau_cents)
        length = sortedCentroids.shape[0]
        leftIndex = int(np.floor(length * 0.16)) - 1
        rightIndex = int(np.ceil(length * 0.84)) - 1
        leftError = sortedCentroids[leftIndex]
        rightError = sortedCentroids[rightIndex]
        tau_cent_err = np.asarray([leftError, rightError])
        return (tau_cent, tau_cent_err)
    
    def getTauPeak(self, correlation, tau_peaks, **kwargs):
        r_max = np.amax(correlation[:,1])
        tau_peak = round(correlation[np.argmax(correlation[:,1]),0])
        number = np.sum(tau_peaks[:,1])
        counter = 0
        for i in range(tau_peaks.shape[1]):
            counter += tau_peaks[i,1]
            if counter >= number /2:
                tau_peak = tau_peaks[i,0]
                break
        length = np.sum(tau_peaks[:,1])
        leftIndex = int(np.floor(length * 0.16)) - 1
        rightIndex = int(np.ceil(length * 0.84)) - 1
        rollingSum = 0
        for i in range(tau_peaks.shape[0]):
            rollingSum += tau_peaks[i,1]
            if rollingSum >= leftIndex:
                leftError = tau_peaks[i,0]
                break
        rollingSum = 0
        for i in range(tau_peaks.shape[0]):
            rollingSum += tau_peaks[i,1]
            if rollingSum >= rightIndex:
                rightError = tau_peaks[i,0]
                break
        tau_peak_err = np.asarray([leftError, rightError]) 
        return (r_max, tau_peak, tau_peak_err)
    
    def resizeSI(self, tau, tau_err):
        self.c = 299792.458 #km/s
        tau_secs = tau * 24 * 3600
        tau_err_secs = tau_err * 24 * 3600
        R = self.c * tau_secs
        R_err = self.c * tau_err_secs
        return (R, R_err)

        
    def calcBHMass(self, which):
        self.f = 1.8 #(Graham, Peterson)
        self.G = 6.67430e-20 #km^3 / kg s^2
        if which == 'cent':
            R = self.R_cent
            R_err = self.R_cent_err
        else:
            R = self.R_peak
            R_err = self.R_peak_err
        
        self.M_kg = self.f * R * self.FWHM_rms**2 / self.G #kg
        self.M_Mo = self.M_kg / 1.988e30 / 1e7
        
        def errorEst(self):
            self.M_err_min = np.sqrt((self.f * R_err[0] * (self.FWHM_rms)**2 / self.G)**2 + 
                                     (self.f * R * 2 * self.FWHM_rms * self.FWHM_rms_errs)**2)
            self.M_err_max = np.sqrt((self.f * R_err[1] * (self.FWHM_rms)**2 / self.G)**2 + 
                                     (self.f * R * 2 * self.FWHM_rms * self.FWHM_rms_errs)**2)
            
        self.M_min_kg = self.f * (R_err[0]) * (self.FWHM_rms - self.FWHM_rms_errs)**2 / self.G #kg
        self.M_min_Mo = self.M_min_kg / 1.988e30 / 1e7
        self.M_max_kg = self.f * (R_err[1]) * (self.FWHM_rms + self.FWHM_rms_errs)**2 / self.G #kg
        self.M_max_Mo = self.M_max_kg / 1.988e30 / 1e7
        self.M_Mo_err = [self.M_min_Mo, self.M_max_Mo]
        
    def getValues(self):
        values = [self.tau_cent, self.tau_cent_err, self.tau_peak, self.tau_peak_err, self.M_Mo]
        return values
        
        
def printTable(filename, linelist):    
    outfile = open(filename, 'w')
    outfile.write(r'\begin{table}[!htb] '+'\n')
    outfile.write(r'\centering '+'\n')
    outfile.write(r'\caption{Cross-correlation lags of the integrated line curves of the Balmer lines as well as \ion{He}{i}\,$\lambda5876$ and \ion{He}{ii}\,$\lambda4686$ with respect to the continua at 4265\,{\AA} and 5100\,{\AA} (at 4425\,{\AA} and 5290\,{\AA} in rest frame, respectively).} '+'\n')
    outfile.write(r'\begin{tabular}{lrrrcrrrr} '+'\n')
    outfile.write(r'\hline '+'\n')
    outfile.write(r'\hline '+'\n')
    outfile.write(r'\multicolumn{1}{c}{Line} & \multicolumn{3}{c}{Cont.\,4265} & & \multicolumn{3}{c}{Cont.\,5100} &  ')
    outfile.write(r' \\')
    outfile.write(' \n')
    outfile.write(r'\cline{2-4}')
    outfile.write(' \n')
    outfile.write(r'\cline{6-8}')
    outfile.write(' \n')
    outfile.write(r'   & \multicolumn{1}{c}{$\tau_{\text{cent}}$} & \multicolumn{1}{c}{$\tau_{\text{cent}}$} & \multicolumn{1}{c}{$\tau_{\text{peak}}$} & & \multicolumn{1}{c}{$\tau_{\text{cent}}$} & \multicolumn{1}{c}{$\tau_{\text{cent}}$} & \multicolumn{1}{c}{$\tau_{\text{peak}}$} & \multicolumn{1}{c}{$M_{\text{BH, FWHM}}$}')
    outfile.write(r' \\')
    outfile.write(' \n')
    outfile.write(' & & \multicolumn{1}{c}{[d]} & \multicolumn{1}{c}{[d]} & & & \multicolumn{1}{c}{[d]} & \multicolumn{1}{c}{[d]} & \multicolumn{1}{c}{[$10^7 M_{\odot}$]}')
    outfile.write(r' \\')
    outfile.write(' \n')
    outfile.write(' \multicolumn{1}{c}{(1)} & \multicolumn{1}{c}{(2)} & \multicolumn{1}{c}{(3)} & \multicolumn{1}{c}{(4)} & & \multicolumn{1}{c}{(5)} & \multicolumn{1}{c}{(6)} & \multicolumn{1}{c}{(7)} & \multicolumn{1}{c}{(8)}')
    outfile.write(r'\\ ')
    outfile.write('\n') 
    outfile.write(r'\hline '+'\n')
    
    for line in linelist:
        outfile.write(line.name+' & $'
                      +str(line.r_max)+'$ & $'
                      +str(line.tau_cent)+'_{'+str(round(line.tau_cent_err[0]-line.tau_cent,1))+'}^{+'+str(round(line.tau_cent_err[1]-line.tau_cent,1))+'}$ & $'
                      +str(line.tau_peak)+'_{'+str(round(line.tau_peak_err[0]-line.tau_peak))+'}^{+'+str(round(line.tau_peak_err[1]-line.tau_peak))+'}$ &'
                      +str(round(line.M_Mo,1))+'_{'+str(round(line.M_Mo_err[0]-line.M_Mo,1))+'}^{+'+str(round(line.M_Mo_err[1]-line.M_Mo,1))+'}$')
        outfile.write(r'\\ ')
        outfile.write('\n')
    
    
    outfile.write(r'\hline '+'\n')
    outfile.write(r'\label{tab:lags_and_masses} '+'\n')
    outfile.write(r'\end{tabular} '+'\n')
    outfile.write(r'\flushleft '+'\n')
    outfile.write(r'\begin{tablenotes} '+'\n')
    outfile.write(r'\footnotesize '+'\n')
    outfile.write(r'\textbf{Notes:} None'+'. \n')
    outfile.write(r'\end{tablenotes} '+'\n')
    outfile.write(r'\end{table}')

"""
lineCorrelations = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/lineCorrelations_ICCF_Hg_Hb_HeI.txt')

    
Ha_cents = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/calculatedCentroidsCont4265_Halpha_ICCF.txt')
Hb_cents = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/calculatedCentroidsCont4265_Hbeta_ICCF.txt')
Hg_cents = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/calculatedCentroidsCont4265_Hgamma_ICCF.txt')
HeI5876_cents = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/calculatedCentroidsCont4265_HeI5876_ICCF.txt')
HeII4686_cents = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/calculatedCentroidsCont4265_HeII4686_ICCF.txt')

Ha_peaks = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/peakDistribution_Cont4265_Halpha_ICCF.txt')
Hb_peaks = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/peakDistribution_Cont4265_Hbeta_ICCF.txt')
Hg_peaks = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/peakDistribution_Cont4265_Hgamma_ICCF.txt')
HeI5876_peaks = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/peakDistribution_Cont4265_HeI5876_ICCF.txt')
HeII4686_peaks = np.loadtxt('../../analysis_noConvolve/1DCorrelations/Cont4265/peakDistribution_Cont4265_HeII4686_ICCF.txt')



Ha = Line(r'H$\alpha$', 1410, 220, 
          np.vstack((lineCorrelations[:,0], lineCorrelations[:,12])).T, Ha_cents, Ha_peaks)
Hb = Line(r'H$\beta$', 1450, 270,
          np.vstack((lineCorrelations[:,0], lineCorrelations[:,4])).T, Hb_cents, Hb_peaks)

HeI5876 = Line('\ion{He}{i}\,$\lambda5876$', 1480, 230, 
               np.vstack((lineCorrelations[:,0], lineCorrelations[:,8])).T, HeI5876_cents, HeI5876_peaks)
HeII4686 = Line('\ion{He}{ii}\,$\lambda4686$', 4800, 900, 
                np.vstack((lineCorrelations[:,0], lineCorrelations[:,3])).T, HeII4686_cents, HeII4686_peaks)




printTable('CCF_lags_and_FWHM_BH_Mass.txt', [Ha, Hb, HeI5876, HeII4686])

"""