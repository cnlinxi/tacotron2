#!/bin/bash

for file in *.wav; do
    c=${file}
    echo $c
    # sox $c -r 22050 -b 16 -c 1 new_$c
    sox $c new_$c tempo -s 0.80
    rm -rf $c
    mv new_$c $c
done