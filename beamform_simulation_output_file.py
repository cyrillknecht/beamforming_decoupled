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
import os
import argparse
import numpy as np
import cv2
import time
import h5py
from scipy.interpolate import interp1d

import sys
from os.path import dirname as up



from beamformer import Beamformer
from simulation_parameters import SimulationParameters

path_to_lib = up(up(up(os.path.realpath(__file__))))
sys.path.insert(0, path_to_lib)

BEAMFORMED_IMG_PREFIX = "beamformed_img_"
BEAMFORMED_IMG_NO_LOG_PREFIX = "beamformed_img_no_log_"
RX_PREFIX = "sensor_data_"
RX_PREFIX_TGC = "sensor_data_tgc_"
PHANTOM_PREFIX = "phantom_"
SIMULATION_INPUT_PREFIX = "simulation_input_file_"
SIMULATION_OUTPUT_PREFIX = "simulation_output_file_"

EVALUATION_METRICS_FILENAME = "metrics.json"
SAVE_BEAMFORM_CLIP = False

def aggregate_over_channels(sensor_data, simulation_parameters: SimulationParameters):
    element_width_points = simulation_parameters.element_width_points
    start_indices = np.arange(0, simulation_parameters.transducer_elements_nr * element_width_points, element_width_points)
    end_indices = start_indices + element_width_points

    aggregated_data = np.array([np.mean(sensor_data[:, start:end], axis=1) for start, end in zip(start_indices, end_indices)]).T
    return aggregated_data

def resample_signal_to_transducer_frq(sensor_data, simulation_parameters: SimulationParameters, dt):
    sampling_freq =  1 / dt
    Nt = sensor_data.shape[0]

    sampling_ratio = simulation_parameters.sampling_frequency / sampling_freq
    Nt_new = round(Nt * sampling_ratio)

    original_indices = np.arange(0, Nt)
    new_indices = np.linspace(0, Nt, Nt_new)

    interp_func = interp1d(original_indices, sensor_data, axis=0, kind='linear', fill_value='extrapolate')
    sensor_data_interpolated = interp_func(new_indices)
    return sensor_data_interpolated

def load_data_simulation_output_file(simulation_output_path, simulation_parameters, save_path_rf):
    sensor_filename = os.path.splitext(os.path.basename(simulation_output_path))[0]
    identifier = sensor_filename[len(SIMULATION_OUTPUT_PREFIX):]
    # load data
    with h5py.File(simulation_output_path, 'r') as hf:
        sensor_data = hf['p'][:]

        sensor_data = sensor_data[0,:,:]
    
        # Load 'dt' field
        dt = hf['dt'][:].item()

    # aggregate channels
    sensor_data_agg = aggregate_over_channels(sensor_data, simulation_parameters)

    # interpolate on other sampling frequency
    sensor_data_resampled = resample_signal_to_transducer_frq(sensor_data_agg, simulation_parameters, dt)
    
    # Save this as Rx data in numpy file
    rf_data_path = os.path.join(save_path_rf, f"{RX_PREFIX}{identifier}.npy")
    np.save(rf_data_path, sensor_data_resampled.astype(np.float32))
    return sensor_data_resampled, identifier


def load_extracted_rf_data(rf_data_path):
    sensor_filename = os.path.splitext(os.path.basename(rf_data_path))[0]
    identifier = sensor_filename[len(RX_PREFIX):]
    rf_data = np.load(rf_data_path)
    return rf_data, identifier

def beamform_single_rf_data(simulation_output_path, 
                            simulation_parameters,
                            save_path_rf,
                            tgc_correction,
                            save_path_tgc_corr_data,
                            save_path_beamformed,
                            store_beamformed_signal,
                            debug,
                            id):
    
    if os.path.splitext(simulation_output_path)[1].lower() == ".h5":
        sensor_data, identifier = load_data_simulation_output_file(simulation_output_path, simulation_parameters, save_path_rf)
    elif os.path.splitext(simulation_output_path)[1].lower() == ".npy":
        sensor_data, identifier = load_extracted_rf_data(simulation_output_path)
    else:
        print("Error: Input file type not supported.")
    if id:
        identifier = id

    # initialize beamformer
    beamformer = Beamformer(simulation_parameters, save_path_beamformed, identifier, sensor_data, debug=debug)

    # Save clipped sensor data
    if SAVE_BEAMFORM_CLIP:
        clipped_sensor_data = beamformer.sensor_data[:,:,0]
        np.save(os.path.join(save_path_rf, f"clipped_sensor_data_{identifier}.npy"), clipped_sensor_data)

    # beamforming
    beamformer.beamform(tgc_gain_correction=tgc_correction, filter=True, store_beamformed_signal=store_beamformed_signal)
    beamformer.image_formation()
    beamformed_image = beamformer.beamformed_image
    beamformed_image_no_log = beamformer.beamformed_image_no_log

    # save tgc corrected signal
    if tgc_correction:
        tgc_corrected_signal_rf_data = beamformer.tgc_corrected_signal
        tgc_corr_cpath = os.path.join(save_path_tgc_corr_data, f"{RX_PREFIX_TGC}{identifier}.npy")
        np.save(tgc_corr_cpath, np.squeeze(tgc_corrected_signal_rf_data).astype(np.float32))

    # Save beamformed image to save_path
    img_path = os.path.join(save_path_beamformed, f"{BEAMFORMED_IMG_PREFIX}{identifier}.png")
    cv2.imwrite(img_path, beamformed_image)
    
    print(f"Beamformed image saved at {img_path}")
    print(f"Beamformed image shape after beamforming: {beamformed_image.shape}")

    no_log_path = os.path.join(save_path_beamformed, f"{BEAMFORMED_IMG_NO_LOG_PREFIX}{identifier}.npy")
    np.save(no_log_path, beamformed_image_no_log.astype(np.float32))

def main():
    args = read_args()

    simulation_setup_file_path = args.parameters_path
    simulation_output_path = args.simulation_output_file
    save_path_beamformed = args.save_path_beamformed
    save_path_rf = args.save_path_rf_data
    save_path_tgc_corr_data = args.save_path_tgc_corr
    debug = args.debug
    id = args.id
    store_beamformed_signal = args.store_beamformed_signal
    skip_tgc = args.skip_tgc
    tgc_correction = not skip_tgc

    # Make sure all save directories exist
    os.makedirs(save_path_beamformed, exist_ok=True)
    os.makedirs(save_path_rf, exist_ok=True)
    os.makedirs(save_path_tgc_corr_data, exist_ok=True)

    simulation_parameters = SimulationParameters(simulation_setup_file_path)   

    print(f"Beamforming simulation output file: {simulation_output_path}")
    if os.path.isfile(simulation_output_path):
        beamform_single_rf_data(simulation_output_path,
                                simulation_parameters,
                                save_path_rf,
                                tgc_correction,
                                save_path_tgc_corr_data,
                                save_path_beamformed,
                                store_beamformed_signal,
                                debug,
                                id)
    elif os.path.isdir(simulation_output_path):
        for filename in os.listdir(simulation_output_path):
            file_path = os.path.join(simulation_output_path, filename)
            name = os.path.splitext(filename)[0]
            identifier = name[len(RX_PREFIX):]
            beamform_single_rf_data(file_path, 
                                    simulation_parameters,
                                    save_path_rf,
                                    tgc_correction,
                                    save_path_tgc_corr_data,
                                    save_path_beamformed,
                                    store_beamformed_signal,
                                    debug,
                                    identifier)
    

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters_path',
                        type=str,
                        required=True,
                        help="Path to simulation config file")
    parser.add_argument('--simulation_output_file',
                        type=str,
                        required=True,
                        help="Path to simulation setup file")
    parser.add_argument('--save_path_rf_data',
                        type=str,
                        required=True,
                        help="Path to folder where we save the rf data")
    parser.add_argument('--save_path_tgc_corr',
                        type=str,
                        required=True,
                        help="Path to folder where we save the tgc corrected signal")
    parser.add_argument('--save_path_beamformed',
                        type=str,
                        required=True,
                        help="Path to folder where we save the beamformed image")
    parser.add_argument('--id',
                        type=str,
                        required=False,
                        help="identifier, if not provided one will be chosen based on the input file name")
    parser.add_argument('--store_beamformed_signal', action='store_true')
    parser.add_argument('--skip_tgc', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    print("Start beamforming")
    time_total_start = time.time()
    main()
    time_total = time.time() - time_total_start
    print(f"Beamform total time {time_total}s")