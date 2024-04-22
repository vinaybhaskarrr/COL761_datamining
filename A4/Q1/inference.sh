#!/bin/bash

if [ "$1" == "test" ]; then
    python3 GC_test.py $2 $3 $4
else
    python3 GC_train.py $2 $3
fi