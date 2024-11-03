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

import sys
import os

from os.path import dirname as up

path_to_lib = up(up(os.path.realpath(__file__)))
sys.path.insert(0, path_to_lib)

import numpy as np
import json

def load_json_file(file_path):
    # Open and read the JSON file for simulation parameters
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def save_json_file(save_path, data):
    data = dict_convert_numbers_to_primitives(data)
    out_json = json.dumps(data, indent=4)
    with open(save_path, 'w') as out_file:
        out_file.write(out_json)

class SimulationParameters():
    def __init__(self, simulation_parameters_path):
        simulation_parameters_dict = load_json_file(simulation_parameters_path)

        self.grid_Nz = simulation_parameters_dict["grid"]["Nz"]
        self.grid_Nx = simulation_parameters_dict["grid"]["Nx"]

        self.axial_cutoff = simulation_parameters_dict["grid"].get("axial_cutoff", [0,0])
       
        self.speed_of_sound  = simulation_parameters_dict["medium"]["mean_sound_speed"]
        self.attenuation_coeff = simulation_parameters_dict["medium"]["alpha_coeff"]
        self.attenuation_power = simulation_parameters_dict["medium"]["alpha_power"]
        self.background_density = simulation_parameters_dict["medium"]["background_density"]

        self.scattering_density_resolution_cell = simulation_parameters_dict["medium"]["scattering_density_resolution_cell"]
        self.resolution_cell_z = simulation_parameters_dict["medium"]["resolution_cell_size"]["axial"]
        self.resolution_cell_x = simulation_parameters_dict["medium"]["resolution_cell_size"]["lateral"]
        self.background_scattering_mean_factor = simulation_parameters_dict["medium"]["background_scattering_mean_factor"]
        self.background_scattering_var_factor = simulation_parameters_dict["medium"]["background_scattering_variance_factor"]

        self.center_frequency = simulation_parameters_dict["transducer"]["center_frequency_hz"]
        self.transducer_elements_nr = simulation_parameters_dict["transducer"]["transducer_elements_nr"]
        self.element_width_points = simulation_parameters_dict["transducer"]["element_width_points"]
        self.element_distance_points = simulation_parameters_dict["transducer"]["element_distance_points"]
        self.transducer_bandwidth = np.array(simulation_parameters_dict["transducer"]["bandwidth"])
        self.samples_per_wavelength = simulation_parameters_dict["transducer"]["samples_per_wavelength"]
        self.input_signal_length_wavelength = simulation_parameters_dict["transducer"]["input_signal_length_wavelengths"]
        self.elevation_focus = simulation_parameters_dict["transducer"]["elevation_focus"]
        self.tgc_control_points = np.array(simulation_parameters_dict["transducer"]["tgc_control_points"])[np.newaxis, :]

        if "tgc_waveform" in simulation_parameters_dict["transducer"]:
            self.tgc_wave_from = np.array(simulation_parameters_dict["transducer"]["tgc_waveform"])[np.newaxis, :]
        else:
            self.tgc_wave_from = self.tgc_control_points

        self.wavelength = self.speed_of_sound / self.center_frequency
        self.element_pitch_points = self.element_width_points + self.element_distance_points 

        if "dxz_wavelength_ratio" in simulation_parameters_dict["grid"]:
            self.dxz_wavelength_ratio = simulation_parameters_dict["grid"]["dxz_wavelength_ratio"]
            self.grid_dxz = self.wavelength / self.dxz_wavelength_ratio
            self.element_pitch = self.element_pitch_points * self.grid_dxz
        else:
            assert "element_pitch_m" in simulation_parameters_dict["transducer"]
            self.element_pitch = simulation_parameters_dict["transducer"]["element_pitch_m"]
            self.grid_dxz = self.element_pitch / self.element_pitch_points
            self.dxz_wavelength_ratio = self.wavelength / self.grid_dxz

        self.height = self.grid_dxz * self.grid_Nz
        self.width = self.grid_dxz * self.grid_Nx
        self.max_depth_wavelength = self.height / self.wavelength
        self.adc_ratio = self.samples_per_wavelength

        self.z_spacing_grid = self.grid_dxz
        self.z_spacing_samples_timesignal = self.wavelength / self.samples_per_wavelength

        self.sampling_frequency = self.samples_per_wavelength * self.center_frequency

        self.axial_cutoff_wavelength = self.input_signal_length_wavelength

        self.z_coord_mm = np.linspace(0, self.height, self.grid_Nz)
        self.x_coord_grid_mm = np.linspace((self.element_pitch * self.transducer_elements_nr / 2) * -1,
                                          (self.element_pitch * self.transducer_elements_nr / 2),
                                          self.transducer_elements_nr*self.element_pitch_points)
        self.x_coord_channel_mm = np.linspace((self.element_pitch * self.transducer_elements_nr / 2) * -1,
                                          (self.element_pitch * self.transducer_elements_nr / 2),
                                          self.transducer_elements_nr)

    def set_grid_Nx(self, new_gridNx):
        self.grid_Nx = new_gridNx
        self.width = self.grid_dxz * self.grid_Nx

    def set_grid_Nz(self, newgridNz):
        self.grid_Nz = newgridNz
        self.height = self.grid_dxz * self.grid_Nz

    def to_dict(self):
        """
        Convert the parameters of the class to a dictionary of same shape as setup file.
        """
        simulation_parameters_dict = {
            "grid": {
                "Nz": self.grid_Nz,
                "Nx": self.grid_Nx,
            },
            "medium": {
                "mean_sound_speed": self.speed_of_sound,
                "alpha_coeff": self.attenuation_coeff,
                "alpha_power": self.attenuation_power,
                "background_density": self.background_density,
                "scattering_density_resolution_cell": self.scattering_density_resolution_cell,
                "resolution_cell_size": {
                    "axial": self.resolution_cell_z,
                    "lateral": self.resolution_cell_x,
                },
                "background_scattering_mean_factor": self.background_scattering_mean_factor,
                "background_scattering_variance_factor": self.background_scattering_var_factor,
            },
            "transducer": {
                "center_frequency_hz": self.center_frequency,
                "transducer_elements_nr": self.transducer_elements_nr,
                "element_width_points": self.element_width_points,
                "element_distance_points": self.element_distance_points,
                "element_pitch_m": self.element_pitch,
                "bandwidth": list(self.transducer_bandwidth),
                "samples_per_wavelength": self.samples_per_wavelength,
                "input_signal_length_wavelengths": self.input_signal_length_wavelength,
                "elevation_focus": self.elevation_focus,
                "tgc_control_points": list(self.tgc_control_points[0]),
            },
        }
        return simulation_parameters_dict

    def save_parameters_to_json(self, save_path):
        simulation_parameter_dict = self.to_dict()
        save_json_file(save_path, simulation_parameter_dict)
