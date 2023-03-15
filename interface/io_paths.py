{
"plot" : True,
"normalize" : True,
# set the directory path where spectra exist for the run
#"spectra_dir_path" : "inputs/spectra/bf-full-survey-data/",
"spectra_dir_path" : "inputs/spectra/bf-validation/",

#"param_path"  : "/Users/jyoon/Dropbox/research/casper-analysis/input_files/bf-survey-casper-input.csv",
#"param_path"  : "inputs/params/bf-final-casper-kdemove-input.csv",
"param_path"  : "inputs/params/he0017+0055-input.csv",
#"param_path"  : "inputs/params/bf-validation_casper-input.csv",

# set the output directory path and
# "output_name" = "output_dir_path"+"output_file_name" in batch.py
# output_name will be prepended to .pdf and .csv for the spectra fit output files,
# MCMC corner plots and output stellar parameters files.
"output_dir_path"   : "outputs/test/de-move_default-norm-params_iter-4000/",
#"output_dir_path"   : "outputs/test/kde-move_new-norm-params_iter-10000/",
#"output_file_name"  : "bf-final_band-check-off_kde-move_walker-100_stepsize-c2-em2_iter_2000"
#"output_file_name"  : "bf-validation_band-check-off_demove0.8-desnooker0.2_walker-100_stepsize-c2-em2-v2"
#"output_file_name"  : "validation-yoon2020_band-check-off_de-move_walker-100_stepsize-c2-em2-v2"
"output_file_name"  : "he0017+0055_cahk-off_band-check-on_demove0.8-desnooker0.2_walker-100_stepsize-c2-em2"
}
