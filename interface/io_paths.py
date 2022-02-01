{
"plot" : True,
"normalize" : True,
# set the directory path where spectra exist for the run
"spectra_dir_path" : "inputs/spectra/bf-full-survey-data/",
#"spectra_dir_path" : "inputs/spectra/test_spectra/",
#"param_path"  : "inputs/params/bf-full-survey-casper-input.csv",
"param_path"  : "inputs/params/bf-survey-casper-input.csv",
# set the output directory path and
# "output_name" = "output_dir_path"+"output_file_name" in batch.py
# output_name will be prepended to .pdf and .csv for the spectra fit output files,
# MCMC corner plots and output stellar parameters files.
"output_dir_path"   : "outputs/bf_survey/case_1/",
"output_file_name"  : "norm_k1_sm25-35_fmin-70_mcmc-800"
}
