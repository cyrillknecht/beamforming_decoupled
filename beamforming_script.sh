#!/bin/bash

test_results_folder="example_test_result"
pred_sensor_data_path="$test_results_folder/Rx_data"
pred_tgc_corr_data_path="$test_results_folder/Rx_data_tgc_corr"
pred_beamformed_images_path="$test_results_folder/beamformed_images"
simulation_setup_file="simulation_setup.json"

id="simulation_output_file_case-100122_BONE_H-N-UXT_3X3_axial_slice_55_crop_1_.h5"
pred_simulation_file_output_path="$pred_sensor_data_path/$id"
beamforming_script="beamform_simulation_output_file.py"

python "$beamforming_script" \
        --parameters_path "$simulation_setup_file" \
        --simulation_output_file "$pred_simulation_file_output_path" \
        --save_path_beamformed "$pred_beamformed_images_path" \
        --save_path_rf_data "$pred_sensor_data_path"\
        --save_path_tgc_corr "$pred_tgc_corr_data_path"