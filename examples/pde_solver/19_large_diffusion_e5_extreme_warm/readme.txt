1. run check_pickle.py to make sure the bnn file and the cnn file used to warm-started the bnn are located in the same folder.
2. because of the re-structure of the folder path, I need to place "data/" in 19_* 20_* 21_*, as the re-start use the older config file, not the new one.
3. l2 error needs to be updated in main_test_extreme.py.
python main_test_extreme.py large-pentagon-64x64-176k-new-cnn.ini -rf results/2021-10-06T22-29-gpu-cn001-NN-large-pentagon-64x64-176k-new-cnn-x0-B2048-E20000-I25-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..datadiffusionlarge-64x64-pentagon-176k-ckpt9870-e20000-ra-ckpt10180.pickle
will output the results. 
4. python main_test_min_max.py large-pentagon-64x64-176k-new-cnn.ini -rf results/2021-10-06T22-29-gpu-cn001-NN-large-pentagon-64x64-176k-new-cnn-x0-B2048-E20000-I25-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..datadiffusionlarge-64x64-pentagon-176k-ckpt9870-e20000-ra-ckpt10180.pickle
5. python main_train_min_max.py large-pentagon-64x64-176k-new-cnn.ini -rf results/2021-10-06T22-29-gpu-cn001-NN-large-pentagon-64x64-176k-new-cnn-x0-B2048-E20000-I25-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..datadiffusionlarge-64x64-pentagon-176k-ckpt9870-e20000-ra-ckpt10180.pickle
6. get the l2 error on screen
python ../report_l2_error.py stats.csv 
(7.): could be used
python ../not-using/analyze_test_info.py stats.csv


# first cnn

run_all_large_results.sh
use process_sim_log.py to get all the summary
plot_l2_error_all.py

#### final 1 ####
./run_e5_all.sh

-- if this is messed up
-- then do the following
-- python process_csv.py in the cnn_log or bnn_log folder
-- then continue the step 2/3/4/

#merge_large_data
--- python ../process_csv.py

#### final 2 ####
python process_sim_log_e5.py sim_log_2021-10-06T22-*.log

#### final 3 ####
mv *.log *.csv log_files/

#### final 4 ####: require all train data
python plot_l2_error_e5.py
### if only test data
python plot_l2_error_e5_test.py

# 2nd bnn



# todo: save all the pickle files in the run folder to plot the final loss
# remove isolated pixels in the final error plot 
# use the 320 data points test resutls
