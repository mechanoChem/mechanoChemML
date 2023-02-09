#!/bin/bash

#all_runs=(run_1 run_2 run_3 run_4)
#all_runs=(run_1 run_3 run_4)
all_runs=(run_2)
this_root=$PWD
simlog=sim_log.log

# cnn
input_file=large-pentagon-64x64-176k-new-cnn.ini
# bnn
input_file=large-pentagon-64x64-176k-new-bnn-warm-1.ini

main_file=main_train_min_max.py
main_file=main_train_min_max_scale_by_max_l2.py
main_file=main_train_min_max_scale_by_max_l2_physical.py
main_file_train=main_train_min_max_scale_by_max_l2_physical_split.py

traine5()
{
  mkdir -p $this_root/train/
  # not tested yet.
  #all_parts=(part1)
  all_parts=(part1 part2 part3 part4 part5)
  # loop over all the arrays
  for p0 in "${all_parts[@]}" 
  do
    echo "$p0"
    cp $this_root/$main_file_train tmp.py
    sed -i "s/FOLDER/regular\/$p0\//g" tmp.py
    sed -i "s/EDGE/5/g" tmp.py
    python tmp.py $input_file -rf $f0
    #echo "e5-stats-$p0-${f0:8}.csv"
    mv stats.csv "e5-stats-$p0-${f0:8}.csv"
  done
  # check why this part is here.
  echo "e5-stats-${f0:8}.csv" >> $simlog
  python $this_root/report_l2_error.py e5-stats-*-${f0:8}.csv >> $simlog 
  mv e5-stats-*-${f0:8}.csv $this_root/train/
  mv $simlog $this_root/train/
  #echo "train e5"
}

teste5()
{
  mkdir -p $this_root/teste/
  echo "test e5"
  all_edges=(regular extreme)
  # loop over all the arrays
  for e0 in "${all_edges[@]}" 
  do
    #echo "train e4569 $f0"
    cp $this_root/$main_file tmp.py
    sed -i "s/FOLDER/new_test\/$e0\//g" tmp.py
    sed -i "s/EDGE/5/g" tmp.py

    python tmp.py $input_file -rf $f0
    mv stats.csv "e5-stats-$e0-${f0:8}.csv"
    echo "e5-stats-$e0-${f0:8}.csv" >> $simlog
    python $this_root/report_l2_error.py e5-stats-$e0-${f0:8}.csv >> $simlog 
    mv e5-stats-$e0-${f0:8}.csv $this_root/teste
    #exit
  done
  mv $simlog $this_root/teste/
}

# loop over all the arrays
for run0 in "${all_runs[@]}" 
do
  this_run=$this_root/$run0
  cd $this_run
  # find all the files
  IFS=$'\n'
  if [[ "$input_file" == *"cnn"* ]]; then
    echo "It's CNN."
    #restarts=($(ls results/2021-10-06T*.pickle))
    restarts=($(ls results/2021-10-06T*-NN-*.pickle))
  fi
  if [[ "$input_file" == *"bnn"* ]]; then
    echo "It's BNN."
    restarts=($(ls results/2021-12-13T*-BNN-*.pickle))
  fi
  unset IFS
  #printf "%s\n" "${restarts[@]}"
  echo "running ... $run0"

  for f0 in "${restarts[@]}" 
  do
    echo "restart ... $f0"
    simlog=sim_log_${f0:8}.log
    if [[ "$f0" == *pentagon-* ]]; then
      echo "e5"
#      traine5
      teste5
    fi
  done
done
