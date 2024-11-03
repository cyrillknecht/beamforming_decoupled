"""
Copyright (C) 2024 ETH Zurich. All rights reserved.

Author: Maurits Reitsma
Supervision: Christoph Leitner, Yawei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import necessary packages
import math
import os
import numpy as np
from matplotlib import pyplot as plt

import sys
from os.path import dirname as up

path_to_lib = up(up(up(up(os.path.realpath(__file__)))))
sys.path.insert(0, path_to_lib)

# to run dasIT, we need to import the necessary commands from the library
from include.dasIT.dasIT.data.loader import RFDataloader
from include.dasIT.dasIT.features.transducer import transducer
from include.dasIT.dasIT.features.medium import medium
from include.dasIT.dasIT.features.tgc import tg_compensation
from include.dasIT.dasIT.src.delays import planewave_delays
from include.dasIT.dasIT.src.apodization import apodization
from include.dasIT.dasIT.src.das_bf import RXbeamformer
from include.dasIT.dasIT.features.signal import RFfilter, fftsignal, analytic_signal
from include.dasIT.dasIT.features.image import interpolate_bf
from include.dasIT.dasIT.visualization.signal_callback import amp_freq_1channel, amp_1channel
from include.dasIT.dasIT.visualization.image_callback import plot_signal_grid, plot_signal_image
from include.dasIT.dasIT.features.signal import logcompression

from simulation_parameters import SimulationParameters
PLOT_TRUE_COLOR = '#1f77b4'
PLOT_PRED_COLOR = '#ff7f0e'

DB_RANGE = 60
CLIP_END = True

class Beamformer():
    def __init__(self, 
                 simulation_parameters: SimulationParameters, 
                 save_path, identifier, 
                 sensor_data = None, 
                 pinmap_ext=None, 
                 debug=False, 
                 height=None):

        # Define all relevant parameters
        self.debug = debug
        self.save_path = save_path
        self.identifier = identifier

        # Define all relevant simulation parameters. These are all either given or computed in the SimluationParameters class
        self.center_frequency = simulation_parameters.center_frequency
        self.transducer_elements_nr = simulation_parameters.transducer_elements_nr
        self.transducer_bandwidth = simulation_parameters.transducer_bandwidth
        self.samples_per_wavelength = simulation_parameters.samples_per_wavelength
        self.axial_cutoff_wavelength = simulation_parameters.axial_cutoff_wavelength
        self.elevation_focus = simulation_parameters.elevation_focus
        self.tgc_control_points = simulation_parameters.tgc_control_points
        self.tgc_waveform = simulation_parameters.tgc_wave_from
        self.grid_Nz = simulation_parameters.grid_Nz
        self.speed_of_sound  = simulation_parameters.speed_of_sound
        self.attenuation_coeff = simulation_parameters.attenuation_coeff
        self.attenuation_power = simulation_parameters.attenuation_power
        self.grid_dxz = simulation_parameters.grid_dxz
        self.element_pitch_points = simulation_parameters.element_pitch_points
        self.element_pitch = simulation_parameters.element_pitch
        self.adc_ratio = simulation_parameters.adc_ratio # adc ratio is the number of time samples per wavelength, it relates center frequency and sampling frequency
        self.z_spacing_samples_timesignal = simulation_parameters.z_spacing_samples_timesignal

        # If we provide the sensor data here
        if sensor_data is not None:
            # Expand dim of sensor data to fit with dasIT structure where multiple frames were allowed. We only ever use 1 frame.
            self.sensor_data = np.expand_dims(sensor_data, axis=2)      

            # clip sensor data to only include samples up to 2*height of the grid.
            height = simulation_parameters.height
            
            # Maximum number of samples to this height.
            max_samples = int(2 * height / self.z_spacing_samples_timesignal)

            # If we have less samples than this amount, then we pad the signal at the end of the time signals with zeros
            if self.sensor_data.shape[0] < max_samples:
                padding_num = max_samples-self.sensor_data.shape[0]
                pad_width = ((0,padding_num), (0, 0), (0, 0))  
                self.sensor_data = np.pad(self.sensor_data, pad_width, mode='constant', constant_values=0)

            # If we have more samples we crop at the end of the time signals
            if self.sensor_data.shape[0] >= max_samples:
                if self.debug:
                    print(f"sensor_data will be cropped from {self.sensor_data.shape[0]} to {max_samples} samples to fit the grid")
                self.sensor_data = self.sensor_data[:max_samples, :, :]

            self.max_time_samples = self.sensor_data.shape[0]
            self.max_depth_wavelength = self.max_time_samples / self.adc_ratio / 2
        else:
            self.max_depth_wavelength = height * self.grid_dxz / simulation_parameters.wavelength
            
        # Pinmap is there to bring channels in correct order
        self.pinmap = pinmap_ext if pinmap_ext is not None else np.array(range(1, self.transducer_elements_nr+1))

        # DEBUG: print all the transducer arguments
        print(f"Center frequency: {self.center_frequency}")
        print(f"bandwidth_hz: {self.transducer_bandwidth}")
        print(f"adc_ratio: {self.adc_ratio}")
        print(f"transducer_elements_nr: {self.transducer_elements_nr}")
        print(f"element_pitch: {self.element_pitch}")
        print(f"pinmap: {self.pinmap}")
        print(f"elevation_focus: {self.elevation_focus}")
        print(f"axial_cutoff_wavelength: {self.axial_cutoff_wavelength}")
        print(f"speed_of_sound: {self.speed_of_sound}")
        
        # Define transducer
        self.dasIT_transducer = transducer(center_frequency_hz = self.center_frequency,   # [Hz]
                                bandwidth_hz=self.transducer_bandwidth,    # [Hz]
                                adc_ratio=self.adc_ratio,  # [-]
                                transducer_elements_nr = self.transducer_elements_nr, # [#]
                                element_pitch_m = self.element_pitch, # [m]
                                pinmap=self.pinmap,
                                pinmapbase=1, # [-]
                                elevation_focus=self.elevation_focus, # [m]
                                focus_number=None,
                                totalnr_planewaves=1,     # [-]
                                planewave_angle_interval=[0,0],   # [rad]
                                axial_cutoff_wavelength=self.axial_cutoff_wavelength,  # [#]
                                speed_of_sound_ms = self.speed_of_sound)  # [m/s]

        # DEBUG: Print all the medium arguments
        print(f"speed_of_sound_ms: {self.dasIT_transducer._speed_of_sound}")
        print(f"center_frequency: {self.dasIT_transducer.center_frequency}")
        print(f"sampling_frequency: {self.dasIT_transducer.sampling_frequency}")
        print(f"max_depth_wavelength: {self.max_depth_wavelength}")
        print(f"lat_transducer_element_spacing: {self.dasIT_transducer.lateral_transducer_spacing}")
        print(f"axial_extrapolation_coef: 1.05")
        print(f"attenuation_coefficient: {self.attenuation_coeff}")
        print(f"attenuation_power: {self.attenuation_power}")
        
        
        # Define medium
        self.dasIT_medium = medium(speed_of_sound_ms = self.dasIT_transducer._speed_of_sound, # [m/s]
                            center_frequency = self.dasIT_transducer.center_frequency, # [Hz]
                            sampling_frequency = self.dasIT_transducer.sampling_frequency, # [Hz]
                            max_depth_wavelength = self.max_depth_wavelength,   # [#]
                            lateral_transducer_element_spacing = self.dasIT_transducer.lateral_transducer_spacing, # [m]
                            axial_extrapolation_coef = 1.05,  # [-]
                            attenuation_coefficient = self.attenuation_coeff,   # [dB/(MHz^y cm)]
                            attenuation_power = self.attenuation_power   # [-]
                            )
        
        if sensor_data is not None:
            # Remove input signal from received signal
            old_sensor_data = np.copy(self.sensor_data)
            
            print(f"Shape of old sensor data: {self.sensor_data.shape}")
            self.sensor_data[:self.dasIT_transducer.start_depth_rec_samples, :, :] = 0
            
            print(f"Shape of sensor data: {self.sensor_data.shape}")

            # Remove final non visible part from received signal
            if CLIP_END:
                clip_off_samples = int(np.ceil(self.dasIT_transducer.start_depth_rec_samples * 1.5))
                self.sensor_data[-clip_off_samples:, :, :] = 0
            else:
                clip_off_samples = 0

            print(f"Cut off RF signal in front and end: [{self.dasIT_transducer.start_depth_rec_samples},{clip_off_samples}]")

            if debug:
                clips = [self.dasIT_transducer.start_depth_rec_samples, self.sensor_data.shape[0]-self.dasIT_transducer.start_depth_rec_samples]
                self.plot_comparison_clip(self.sensor_data, old_sensor_data, clips)
        if self.debug:
            print(f"Height of medium in # of points: {self.dasIT_medium.medium[1].size}")

    def set_beamformed_signal(self, beamformed_signal):
        self.beamformed_signal = np.expand_dims(beamformed_signal, axis=2)

    def beamform(self, tgc_gain_correction=True, filter=True, store_beamformed_signal= False):
        # Plot received sensor data
        if self.debug:
            self.plot_sensor_data(self.sensor_data) 
            self.plot_multiple_RF_channels(self.sensor_data)    
            print("Beamforming")

        ####################################################################
        # ----------------------- Time gain compenation -----------------------

        if tgc_gain_correction:
            print("apply tgc")

            self.tgc_corrected_signal = self.apply_tgc_correction(self.sensor_data)
            next_signal = self.tgc_corrected_signal 
            if self.debug:
                self.plot_sensor_data(next_signal, "tgc_corr") 
                self.plot_tgc_comparison()  
        else: 
            next_signal = self.sensor_data

        ####################################################################
        # ----------------------- Gaussian Bandpass filter -----------------------

        if filter:

            print("Filter signal")
        
            self.RF_filtered = RFfilter(signals=next_signal,
                                    fcutoff_band=self.dasIT_transducer.bandwidth,
                                    fsampling=self.dasIT_transducer.sampling_frequency,
                                    type='gaussian',
                                    order=100)
            next_signal = self.RF_filtered.signal

        
        ####################################################################
        #------------------------ Analytical Signal -----------------------#

        print("Envelope detection")

        ### Hilbert Transform
        self.RFdata_analytic = analytic_signal(np.squeeze(next_signal), interp=False)
        if len(self.RFdata_analytic.shape)==2:
            self.RFdata_analytic = np.expand_dims(self.RFdata_analytic, 2)

        ####################################################################
        #------------------------ Plot processing steps-----------------------#
        if self.debug:
            self.plot_processing_steps(filter, tgc_gain_correction)
            
        ####################################################################
        #-------------------------- Apodization Table --------------------------#

        apodization_rec = apodization(delays=None,
                                medium=self.dasIT_medium.medium,
                                transducer=self.dasIT_transducer,
                                apo='hanning',
                                angles=self.dasIT_transducer.planewave_angles())


        ####################################################################
        #-------------------------- Delay Tables --------------------------#


        ### DAS delay tabels for tilted planewaves
        delay_table = planewave_delays(medium=self.dasIT_medium.medium,
                                    sos=self.dasIT_medium.speed_of_sound,
                                    fsampling=self.dasIT_transducer.sampling_frequency,
                                    angles=self.dasIT_transducer.planewave_angles())
        if self.debug:
            print(self.dasIT_medium.rx_echo_totalnr_samples)
            print(f"Shape of delay table: {delay_table.sample_delays.shape}")
            print(np.max(delay_table.sample_delays))


        ####################################################################
        #-------------------------- Beamforming ---------------------------#

        # Mask images areas in axial direction which have been included for reconstruction
        # but are not part of the actual image.
        RFsignals = self.RFdata_analytic[:,:,0]

        RFsignals = np.expand_dims(RFsignals, 2)
        RFsignals = np.repeat(RFsignals, RFsignals.shape[1], axis=2)
        RFsignals = np.expand_dims(RFsignals, 3)

        self.BFsignals = RXbeamformer(signals=RFsignals,
                                delays=delay_table.sample_delays,
                                apodization=apodization_rec.table)
        
        self.beamformed_signal =  self.BFsignals.frame

        if self.debug:
            print(f"Shape of beamformed image: {self.beamformed_signal.shape}")

        self.beamformed_signal = np.expand_dims(self.beamformed_signal, axis=2)

        if store_beamformed_signal:
            beamformed_signal_save_path = os.path.join(self.save_path, f"beamformed_signal_{self.identifier}.npy")
            np.save(beamformed_signal_save_path, abs(self.beamformed_signal).astype(np.float32))

    def get_tgc_corrected_signal(self):
        signal = self.sensor_data
        RF_TGCsignals = tg_compensation(signals=signal,
                                        medium=self.dasIT_medium,
                                        center_frequency=self.dasIT_transducer.center_frequency,
                                        cntrl_points=self.tgc_waveform,
                                        mode='points')
        
        return RF_TGCsignals.signals
    
    def apply_tgc_correction(self, signal):
        self.RF_TGCsignals = tg_compensation(signals=signal,
                                        medium=self.dasIT_medium,
                                        center_frequency=self.dasIT_transducer.center_frequency,
                                        cntrl_points=self.tgc_waveform,
                                        mode='points')
        
        return self.RF_TGCsignals.signals

    def image_formation(self, plot_output=False, filter = False, hilbert=False):

        next_signal = self.beamformed_signal

        if filter:
            print("Filter signal")
        
            self.RF_filtered = RFfilter(signals=next_signal,
                                    fcutoff_band=self.dasIT_transducer.bandwidth,
                                    fsampling=self.dasIT_transducer.sampling_frequency,
                                    type='gaussian',
                                    order=100)
            next_signal = self.RF_filtered.signal

        if hilbert:

            print("Envelope detection")

            ### Hilbert Transform
            self.RFdata_analytic = analytic_signal(np.squeeze(next_signal), interp=False)
            if len(self.RFdata_analytic.shape)==2:
                self.RFdata_analytic = np.expand_dims(self.RFdata_analytic, 2)

            next_signal = self.RFdata_analytic

        ####################################################################
        #------------------------ Interpolation of Image, both axially and laterally --------------------------

        print("Image Formation")

        # Envelope
        self.BF_signals_envelope = abs(next_signal)

        # This beamformed signal now has axial pixel distance of temporal_resolution / speed_of_sound. 
        # Depending on how you choose the grid spacing, the axial pixel distance this might not match
        # Now we interpolate the beamformed image axially such that the pre deterimened axial pixel spacing is used.

        z_spacing_bf_signal = self.dasIT_medium.medium[1][1] - self.dasIT_medium.medium[1][0] # This is grid spacing of current image

        assert math.isclose(z_spacing_bf_signal, self.z_spacing_samples_timesignal, abs_tol=1e-7) # Make sure this matches what we expect

        axial_interpolation_factor = self.z_spacing_samples_timesignal / self.grid_dxz # Interpolation factor to get desired grid spacing

        # Interpolate the signal to the desired axial size
        BF_signals_envelope_on_grid = interpolate_bf(self.BF_signals_envelope, self.dasIT_transducer, self.dasIT_medium, axial_interpolation_factor, lateral_scale=1)
        
        self.BF_signals_envelope = BF_signals_envelope_on_grid.signals_interp

        # If the interpolated signal is larger than the grid size Nz we cut it off at the end to get a grid of size Nz
        if self.BF_signals_envelope.shape[0] > self.grid_Nz:
            self.BF_signals_envelope = self.BF_signals_envelope[:self.grid_Nz, :]
        # assert self.BF_signals_envelope.shape[0] == self.grid_Nz

        # Update the medium object to match the interpolated gird size
        self.dasIT_medium.update_imaging_medium_axially(axial_distance=self.grid_Nz * self.grid_dxz, axial_len=self.grid_Nz)

        if self.debug:
            self.plot_non_interpolated_beamformed_image()

        # Interpolate over Lateral space, such that lateral pixel size grid size.
        # Lateral space now is n_channels, want to get to Nx
        lateral_interpolation_scale = self.element_pitch_points
        axial_interpolation_scale = 1
        self.BF_interpolated_signal = interpolate_bf(signals=self.BF_signals_envelope,
                                                transducer=self.dasIT_transducer,
                                                medium=self.dasIT_medium,
                                                lateral_scale=lateral_interpolation_scale,
                                                axial_scale=axial_interpolation_scale)

        axial_clip = [self.dasIT_transducer.start_depth_rec_m, None]

        if plot_output:
            plot_path = os.path.join(self.save_path, f"beamformed_plot_{self.identifier}.png")
        else:
            plot_path = None

        self.beamformed_image = plot_signal_grid(signals=self.BF_interpolated_signal.signals_interp,
                        axis_vectors_xz=self.BF_interpolated_signal.imagegrid_mm,
                        axial_clip=axial_clip,
                        compression=True,
                        dbrange=DB_RANGE,
                        path=plot_path,
                        pad = True)
        
        self.beamformed_image_no_log = plot_signal_grid(signals=self.BF_interpolated_signal.signals_interp,
                        axis_vectors_xz=self.BF_interpolated_signal.imagegrid_mm,
                        axial_clip=axial_clip,
                        compression=False,
                        dbrange=DB_RANGE,
                        pad=True)
        if self.debug:
            print(f"Shape of final beamformed image: {self.beamformed_image.shape}")
            print(f"Lateral pixel size in mm: {self.BF_interpolated_signal.grid_conversion_px2mm[0]}")
            print(f"Axial pixel size in mm: {self.BF_interpolated_signal.grid_conversion_px2mm[1]}")


####################################################################################################
    #------------------------ Plotting Functions --------------------------
    def plot_sensor_data(self, sensor_data, postamble = ""):
        print(f"Expected number of time samples:{self.dasIT_medium.rx_echo_totalnr_samples}")
        print(f"Number of time samples: {self.max_time_samples}")

        signal = logcompression(sensor_data[:,:,0], DB_RANGE)        
        plt.figure(figsize=(5, 10), dpi=400)
        plt.imshow(signal,
                    aspect=1,
                    cmap='gray')

        plt.xlabel('Transducer Element [#]', fontsize=15, fontweight='bold', labelpad=10)
        plt.ylabel('Passing Time (Sample [#])', fontsize=15, fontweight='bold', labelpad=10)
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plot_path = os.path.join(self.save_path, f"rx_data_{postamble}_{self.identifier}.png")
        plt.savefig(plot_path)
        plt.close()

        plt.figure(figsize=(5, 10), dpi=400)
        plt.imshow(signal,
                    aspect=1,
                    cmap='gray')
        plt.axis('off')  # Turn off axes
        plot_path = os.path.join(self.save_path, f"rx_data_{postamble}_{self.identifier}_tight.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_multiple_RF_channels(self, sensor_data):
        num_plots = 3 
        x_min, x_max = 0, sensor_data.shape[0]
        channels = [i for i in range(num_plots)]
        plot_path = os.path.join(self.save_path, f"multiple_rf_channels_{self.identifier}.svg")

        # Creating a figure with subplots
        fig, axs = plt.subplots(num_plots, 1, figsize=(15, 2 * num_plots), gridspec_kw={'hspace': 0})

        # Check if axs is an array of subplots or a single subplot object
        if num_plots == 1:
            axs = [axs]  # Make it a list to simplify the loop below

        # Plot each specified channel
        for i, channel_nr in enumerate(channels):
            channel_data = sensor_data[:, channel_nr, 0]  # Assuming the data for each channel is in the second dimension
            axs[i].plot(channel_data, color='blue')  # Define color or use a predefined variable
            axs[i].set_xticks([])  # Remove x-axis ticks
            axs[i].set_yticks([])  # Remove y-axis ticks
            axs[i].grid(False)
            axs[i].set_xlim([x_min, x_max])

        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()


    def plot_tgc_comparison(self):
        channel_nr = self.sensor_data.shape[1] // 2
        orignal_rf_data = self.sensor_data[:, channel_nr, 0]
        tgc_corrected_signal = self.RF_TGCsignals.signals[:,channel_nr]

        x_min, x_max = 0, self.sensor_data.shape[0]
        plot_path = os.path.join(self.save_path, f"compare_tgc_signal_{self.identifier}.svg")

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        plt.rcParams['xtick.labelsize'] = 14  # Sets the x-axis tick font size
        plt.rcParams['ytick.labelsize'] = 14  # Sets the y-axis tick font size
        
        # Original signal 
        axs[0].plot(orignal_rf_data, label=f'Original RF data', color=PLOT_TRUE_COLOR)
        axs[0].set_title(f'Original RF data', fontsize=18)
        axs[0].set_ylabel('Amplitude [Pa]', fontsize=14)
        axs[0].grid(True)
        axs[0].set_xlim([x_min, x_max])
        
        # Clipped signal
        axs[1].plot(tgc_corrected_signal, label='TGC Corrected RF data', color=PLOT_PRED_COLOR)
        axs[1].set_title(f'TGC Corrected RF data', fontsize=18)
        axs[1].set_xlabel('Time sample #', fontsize=14)
        axs[1].set_ylabel('Amplitude [Pa]', fontsize=14)
        axs[1].grid(True)
        axs[1].set_xlim([x_min, x_max])

        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_envelope_detection():
        pass


    def plot_beamformed_signal(self):
        beamformed_signal_compressed = logcompression(self.BFsignals.frame, DB_RANGE)
        beamformed_signal = self.BFsignals.frame

        plt.figure(figsize=(5, 10), dpi=100)
        plt.imshow(beamformed_signal_compressed,
                    aspect=1,
                    cmap='gray')
        plt.xlabel('Transducer Element [#]', fontsize=15, fontweight='bold', labelpad=10)
        plt.ylabel('Depth (Sample [#])', fontsize=15, fontweight='bold', labelpad=10)
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        plot_path = os.path.join(self.save_path, f"beamformed_signal_{self.identifier}.png")
        plt.savefig(plot_path)
        plt.close()

        plt.figure(figsize=(5, 10), dpi=100)
        plt.imshow(beamformed_signal_compressed,
                    aspect=1,
                    cmap='gray')
        plt.axis('off')  # Turn off axes
        plot_path = os.path.join(self.save_path, f"beamformed_signal_{self.identifier}_tight.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.figure(figsize=(5, 10), dpi=100)
        plt.imshow(abs(beamformed_signal),
                    aspect=1,
                    cmap='gray')
        plt.axis('off')  # Turn off axes
        plot_path = os.path.join(self.save_path, f"beamformed_signal_{self.identifier}_no_log_tight.png")
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()


    def plot_processing_steps(self, filter, tgc_gain_correction):
        channel = self.sensor_data.shape[1] // 2
        fft_receivedsignal = fftsignal(self.sensor_data[:,channel,0], self.dasIT_transducer.sampling_frequency)

        if filter:
            fftFil = fftsignal(self.RF_filtered.signal[:,channel,0], self.dasIT_transducer.sampling_frequency)

        total_subplots = 3 + 2*int(filter) + int(tgc_gain_correction)

        fig, axs = plt.subplots(total_subplots, 1, figsize=(6, 9))

        axs[0].plot(self.sensor_data[:,channel,0])
        axs[0].set_xlabel('Samples [#]')
        axs[0].set_ylabel('Signal [V]')
        axs[0].set_title(f'RF-data channel {channel}')

        subplot_count = 1

        if tgc_gain_correction:
            axs[subplot_count].plot(self.RF_TGCsignals.signals[:,channel],'r')
            axs[subplot_count].set_xlabel('Samples [#]')
            axs[subplot_count].set_ylabel('Signal [V]')
            axs[subplot_count].set_title(f'TGC compensated RF-data channel {channel}')
            subplot_count += 1

        if filter:
            axs[subplot_count].plot(self.RF_filtered.signal[:,channel,0],'r')
            axs[subplot_count].set_xlabel('Samples [#]')
            axs[subplot_count].set_ylabel('Signal [V]')
            axs[subplot_count].set_title(f'Filtered RF-data channel {channel}')
            subplot_count += 1

        axs[subplot_count].plot(self.RFdata_analytic[:,channel,0].real,'r')
        axs[subplot_count].plot(abs(self.RFdata_analytic[:,channel,0]),'g', label='envelope')
        axs[subplot_count].set_xlabel('Samples [#]')
        axs[subplot_count].set_ylabel('Signal [V]')
        axs[subplot_count].set_title(f'Analytic signal channel {channel}')
        axs[subplot_count].legend(loc='lower right')
        subplot_count += 1
        
        axs[subplot_count].plot(fft_receivedsignal[0], fft_receivedsignal[1], 'r')
        axs[subplot_count].set_xlabel('Frequency [MHz]')
        axs[subplot_count].set_ylabel('Power [W/Hz]')
        axs[subplot_count].set_title(f'FFT channel {channel}')
        subplot_count += 1

        if filter:
            axs[subplot_count].plot(fftFil[0], fftFil[1], 'r')
            axs[subplot_count].set_xlabel('Frequency [MHz]')
            axs[subplot_count].set_ylabel('Power [W/Hz]')
            axs[subplot_count].set_title(f'FFT filtered channel {channel}')
            subplot_count += 1

        fig.suptitle('Pre-beamforming processing')

        plt.tight_layout()
        plot_processing_step_path = os.path.join(self.save_path, f"processing_steps_{self.identifier}.png")
        plt.savefig(plot_processing_step_path)
        plt.close()

    def plot_comparison_clip(self, cut_sensor_data, original_sensor_data, clips):
        x_min, x_max = 0, cut_sensor_data.shape[0]
        channel_nr = cut_sensor_data.shape[1] // 2
        channel_data_cut = cut_sensor_data[:, channel_nr, 0]
        channel_data_original = original_sensor_data[:, channel_nr, 0]
        plot_path = os.path.join(self.save_path, f"compare_rf_signal_cut_{self.identifier}.png")

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        
        # Original signal 
        axs[0].plot(channel_data_original, label=f'Original signal', color=PLOT_TRUE_COLOR)
        axs[0].set_title(f'Original signal channel {channel_nr}')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)
        axs[0].axvline(x=clips[0], color='black', linestyle='--', alpha=1)
        axs[0].axvline(x=clips[1], color='black', linestyle='--', alpha=1)
        axs[0].axvspan(xmin=0, xmax=clips[0], color='grey', alpha=0.3)
        axs[0].axvspan(xmin=clips[1], xmax=axs[0].get_xlim()[1], color='grey', alpha=0.3)
        axs[0].set_xlim([x_min, x_max])
        
        # Clipped signal
        axs[1].plot(channel_data_cut, label='Clipped signal', color=PLOT_PRED_COLOR)
        axs[1].set_title(f'Clipped signal channel {channel_nr}')
        axs[1].set_xlabel('Time sample #')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)
        axs[1].axvline(x=clips[0], color='black', linestyle='--', alpha=1)
        axs[1].axvline(x=clips[1], color='black', linestyle='--', alpha=1)
        axs[1].axvspan(xmin=0, xmax=clips[0], color='grey', alpha=0.3)
        axs[1].axvspan(xmin=clips[1], xmax=axs[1].get_xlim()[1], color='grey', alpha=0.3)
        axs[1].set_xlim([x_min, x_max])

        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()


    def plot_non_interpolated_beamformed_image(self):
        # Plot Non interpolated image
        bf_NonInterpolated = interpolate_bf(signals=self.BF_signals_envelope,
                    transducer=self.dasIT_transducer,
                    medium=self.dasIT_medium,
                    axial_scale=1,
                    lateral_scale=1)
        
        axis_vectors_xz = bf_NonInterpolated.imagegrid_mm
        signal = logcompression(self.BF_signals_envelope, dbrange=DB_RANGE)
        plot_path = os.path.join(self.save_path, f"non_interpolated_plot_{self.identifier}.png")

        fig = plt.figure(figsize=(5, 7), dpi=300)
        ax_1 = fig.add_subplot(111)
        ax_1.imshow(np.flipud(signal),
                    aspect=1,
                    interpolation='none',
                    origin='upper',
                    cmap='gray')

        ax_1.invert_yaxis()

        ax_1.set_xlabel('Lateral', fontsize=15, fontweight='bold', labelpad=10)
        ax_1.set_ylabel('Axial', fontsize=15, fontweight='bold', labelpad=10)
        ax_1.xaxis.tick_top()
        ax_1.xaxis.set_label_position('top')
        ax_1.minorticks_on()

        lateral_px_mm = axis_vectors_xz[0][1] - axis_vectors_xz[0][0] 
        axial_px_mm = axis_vectors_xz[1][1] - axis_vectors_xz[1][0] 
        ax_1.text(-100, -30, f"Lateral pixel size in mm: {round(lateral_px_mm, 5)}")
        ax_1.text(-100, -50, f"Axial pixel size in mm: {round(axial_px_mm,5)}")
        fig.savefig(plot_path, dpi=300)
