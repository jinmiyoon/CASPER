{
"plot" : True,
"normalize" : True,
# set the directory path where spectra exist for the run
#"spectra_dir_path" : "inputs/spectra/bf-full-survey-data/",
"spectra_dir_path" : "inputs/spectra/bf-validation/",

#"param_path"  : "inputs/params/bf-final-casper-demove0.8_desnooker0.2-input.csv",
#"param_path"  : "inputs/params/bf-final-casper-input.csv",
#"param_path"  : "inputs/params/placco2013.csv",
"param_path"  : "inputs/params/bf-validation_casper-input.csv",
#"param_path"  : "inputs/params/g77-61-input.csv",

# set the output directory path and
# "output_name" = "output_dir_path"+"output_file_name" in batch.py
# output_name will be prepended to .pdf and .csv for the spectra fit output files,
# MCMC corner plots and output stellar parameters files.
#"output_dir_path"   : "outputs/test/de-move_default-norm-params_iter-4000/",
"output_dir_path"   : "outputs/test/synthetic_lib/demove0.8_desnooker0.2/",
#"output_dir_path"   : "outputs/test/kde-move_new-norm-params_iter-10000/",
#"output_file_name"  : "bf-final_band-check-off_kde-move_walker-100_stepsize-c2-em2_iter_2000"
#"output_file_name"  : "bf-validation_band-check-off_demove0.8-desnooker0.2_walker-100_stepsize-c2-em2-v2"
#"output_file_name"  : "validation-yoon2020_band-check-off_de-move_walker-100_stepsize-c2-em2-v2"
#"output_file_name"  : "jy1051_cahk-off_band-check-on_demove0.8-desnooker0.2_walker-100_stepsize-c2-em2"
#"output_file_name"  : "bf-final2_nwalker-64_cahk-True_bandcheck-False_GISIC_updated_bands_synthetic"
"output_file_name"  : "bf-validation2-ch-c2_nwalker-64_cahk-True_bandcheck-False_GISIC_updated_bands_synthetic"
}
