#!/bin/bash


for lr in 1e-2 1e-3
do
  for batch in 32 64
  do
    for nepoch in 100 200
    do
      dirname="lr$lr-b$batch-n$nepoch"
      mkdir $dirname
      cd $dirname
      ln -s ../dataset.pt .
      ln -s ../python_src
      cat ../config.tmpl | sed s/@lr@/$lr/g\
                         | sed s/@batch@/$batch/g\
                         | sed s/@nepoch@/$nepoch/g\
                         > config.txt
      cd ../
    done
  done
done
