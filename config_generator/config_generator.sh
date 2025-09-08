#!/bin/bash


for lr in 1e-2
do
  for batch in 32
  do
    for nepoch in 100 
    do
      dirname="lr$lr-b$batch-n$nepoch"
      mkdir $dirname
      cd $dirname
      cat ../config.tmpl | sed s/@lr@/$lr/g\
                         | sed s/@batch@/$batch/g\
                         | sed s/@nepoch@/$nepoch/g\
                         > config.txt
      ln -s ../python_src .
      cd python_src/bin/
      python3 py1.py >& ../../$dirname/log 
      cd ../../
      cd ../
    done
  done
done
