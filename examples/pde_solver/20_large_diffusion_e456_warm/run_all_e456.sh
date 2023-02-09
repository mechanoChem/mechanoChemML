#!/bin/bash

#all_runs=(run_1 run_2 run_3 run_4 run_5 run_6 run_7)
all_runs=(run_1 run_3 run_4 run_5 run_6 run_7)
all_runs=(run_2)
this_root=$PWD
simlog=sim_log.log

# cnn
input_file=large-e456-64x64-192k-cnn.ini
# bnn
input_file=large-e456-64x64-192k-bnn-warm-1.ini

main_file=main_train_min_max.py
main_file=main_train_min_max_scale_by_max_l2.py
main_file=main_train_min_max_scale_by_max_l2_physical.py
main_file_train=main_train_min_max_scale_by_max_l2_physical_split.py

traine456()
{
  mkdir -p $this_root/train/
  echo "train e456"
  all_edges=(large-64x64-e4-32k large-64x64-e5-64k large-64x64-e6-96k)
  all_parts=(part1 part2 part3 part4 part5)
  # loop over all the arrays
  for e0 in "${all_edges[@]}" 
  do
    for p0 in "${all_parts[@]}" 
    do
      echo "$p0"
      #echo "train e4569 $f0"
      cp $this_root/$main_file_train tmp.py
      sed -i "s/FOLDER/$e0\/$p0\//g" tmp.py

      if [[ "$e0" == *e6-* ]]; then
        sed -i "s/EDGE/6/g" tmp.py
      fi
      if [[ "$e0" == *e5-* ]]; then
        sed -i "s/EDGE/5/g" tmp.py
      fi
      if [[ "$e0" == *e4-* ]]; then
        sed -i "s/EDGE/4/g" tmp.py
      fi
      python tmp.py $input_file -rf $f0
      mv stats.csv "e456-stats-$e0-$p0-${f0:8}.csv"
    done
    echo "e456-stats-$e0-${f0:8}.csv" >> $simlog
    python $this_root/report_l2_error.py e456-stats-$e0*-${f0:8}.csv >> $simlog 
    mv e456-stats-*-${f0:8}.csv $this_root/train/
    mv $simlog $this_root/train/
  done
}

teste456()
{
  mkdir -p $this_root/teste/
echo "test e456"
  all_edges=(s4 s5 s6 s7 s8 s9 s10 s11 s12)
  #all_edges=(s4)
  # loop over all the arrays
  for e0 in "${all_edges[@]}" 
  do
    #echo "train e4569 $f0"
    cp $this_root/$main_file tmp.py
    sed -i "s/FOLDER/new_test\/$e0\//g" tmp.py

    if [[ "$e0" == *s12* ]]; then
      sed -i "s/EDGE/12/g" tmp.py
    fi
    if [[ "$e0" == *s11* ]]; then
      sed -i "s/EDGE/11/g" tmp.py
    fi
    if [[ "$e0" == *s10* ]]; then
      sed -i "s/EDGE/10/g" tmp.py
    fi
    if [[ "$e0" == *s9* ]]; then
      sed -i "s/EDGE/9/g" tmp.py
    fi
    if [[ "$e0" == *s8* ]]; then
      sed -i "s/EDGE/8/g" tmp.py
    fi
    if [[ "$e0" == *s7* ]]; then
      sed -i "s/EDGE/7/g" tmp.py
    fi
    if [[ "$e0" == *s6* ]]; then
      sed -i "s/EDGE/6/g" tmp.py
    fi
    if [[ "$e0" == *s5* ]]; then
      sed -i "s/EDGE/5/g" tmp.py
    fi
    if [[ "$e0" == *s4* ]]; then
      sed -i "s/EDGE/4/g" tmp.py
    fi

    python tmp.py $input_file -rf $f0
    mv stats.csv "e456-stats-$e0-${f0:8}.csv"
    echo "e456-stats-$e0-${f0:8}.csv" >> $simlog
    python $this_root/report_l2_error.py e456-stats-$e0-${f0:8}.csv >> $simlog 
    mv e456-stats-$e0-${f0:8}.csv $this_root/teste
  done
  mv $simlog $this_root/teste/
  #exit
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
    if [[ "$f0" == *e456-* ]]; then
      echo "e456"
      #traine456
      teste456
    fi
  done
done
