#!/bin/bash

# if you have been doing computing for 5 years and face
# the need to run many codes for 5 years. If you have been
# manually editing hundreds of configure files.
# 
# and you have not think of a way to automate this manual
# problem. then you are not spending your time well.


# for look to go through grid search of hyper-parameters
# create one directory for each hyper-parameter setting
for lr in 1e-2
do
  for batch in 32
  do
    for nepoch in 100 
    do
      dirname="lr$lr-b$batch-n$nepoch"
      mkdir $dirname     # create the directory
      cd $dirname
      # configure the config file using regular expression
      cat ../config.tmpl | sed s/@lr@/$lr/g\
                         | sed s/@batch@/$batch/g\
                         | sed s/@nepoch@/$nepoch/g\
                         > config.txt
      ln -s ../python_src .      # symbolic link to link the python source code
      cd python_src/bin/
      # run the python script, output into proper directory
      # also use commandline or other means to read in the correct config file
      python3 py1.py ../../$dirname/config.txt >& ../../$dirname/log  
      cd ../../
      cd ../
    done
  done
done
