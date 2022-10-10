{
"plot" : True,
"normalize" : True,
# set the directory path where spectra exist for the run
"spectra_dir_path" : "inputs/spectra/bf-full-survey-data/",
#"spectra_dir_path" : "inputs/spectra/test_spectra/",

#"param_path"  : "/Users/jyoon/Dropbox/research/casper-analysis/input_files/bf-survey-casper-input.csv",
#"param_path"  : "inputs/params/bf-full-survey-casper-input.csv",
#"param_path"  : "inputs/params/validation-stars-modified-RV-input.csv",
#"param_path"  : "inputs/params/rv-correct-rest-input.csv",
"param_path"  : "inputs/params/jy2552-input.csv",

# set the output directory path and
# "output_name" = "output_dir_path"+"output_file_name" in batch.py
# output_name will be prepended to .pdf and .csv for the spectra fit output files,
# MCMC corner plots and output stellar parameters files.
"output_dir_path"   : "outputs/test/new_norm_params/",
#"output_file_name"  : "bf-full_band-check-on_kde-move_multi_walker-100_stepsize-c2-em2-v2"
"output_file_name"  : "jy2552_band-check-on_kde-move_walker-100_stepsize-c2-em2-v2"
}
