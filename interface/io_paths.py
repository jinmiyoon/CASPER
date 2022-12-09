{
"plot" : True,
"normalize" : True,
# set the directory path where spectra exist for the run
#"spectra_dir_path" : "inputs/spectra/bf-full-survey-data/",
"spectra_dir_path" : "inputs/spectra/validation-yoon2020/",

#"param_path"  : "/Users/jyoon/Dropbox/research/casper-analysis/input_files/bf-survey-casper-input.csv",
#"param_path"  : "inputs/params/bf-full-survey-casper-input.csv",
#"param_path"  : "inputs/params/cs30314-input.csv",
"param_path"  : "inputs/params/validation-yoon2020-input.csv",

# set the output directory path and
# "output_name" = "output_dir_path"+"output_file_name" in batch.py
# output_name will be prepended to .pdf and .csv for the spectra fit output files,
# MCMC corner plots and output stellar parameters files.
#"output_dir_path"   : "outputs/test/kde_move_new_norm_params/",
"output_dir_path"   : "outputs/test/validation_yoon-2020/",
#"output_file_name"  : "bf-full_band-check-off_kde-move_multi_walker-100_stepsize-c2-em2-v2"
#"output_file_name"  : "cs30314_band-check-off_de-move_walker-100_stepsize-c2-em2-v2"
"output_file_name"  : "validation-yoon2020_band-check-off_de-move_walker-100_stepsize-c2-em2-v2"
}
