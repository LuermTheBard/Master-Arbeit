#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:47:15 2024

@author: Malte
"""
import numpy as np

class Line():
    def __init__(self, name, FWHM_rms, FWHM_rms_errs, correlation, tau_cents, tau_peaks, which='cent', **kwargs):
        self.name = name
        self.FWHM_rms = FWHM_rms
        self.FWHM_rms_errs = FWHM_rms_errs
        self.correlation = correlation
        self.tau_cents = tau_cents
        self.tau_peaks = tau_peaks
        self.tau_cent, self.tau_cent_err = self.getTauCent(self.correlation, self.tau_cents)
        self.R_cent, self.R_cent_err = self.resizeSI(self.tau_cent, self.tau_cent_err)
        self.r_max, self.tau_peak, self.tau_peak_err = self.getTauPeak(self.correlation, self.tau_peaks)
        self.R_peak, self.R_peak_err = self.resizeSI(self.tau_peak, self.tau_peak_err)
        self.calcBHMass(which=which)

        
    
    def getTauCent(self, correlation, tau_cents, **kwargs):
        threshold = 0.8 * np.amax(correlation[:,1])
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
        self.f = 1.8 * 3.77  #(Graham, Peterson: 1.8; 3.77 scale to 6.8 due to inclination of NGC4593 of i ~ 11° (Krolik, Ochmann))
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
