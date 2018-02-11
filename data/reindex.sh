#!/bin/bash

function reindex() {
    a=0
    for i in $1/*.jpg; do
        new=$(printf "$1/$1%04d.jpg" "$a") #04 pad to length of 4
        mv -i -- "$i" "$new"
        let a=a+1
    done
}

function reindex_helper() {
    mv $1 $1_tmp
    reindex $1_tmp

    mv $1_tmp $1
    reindex $1
}


for i in "$@"; do
   echo "Reindex: "$i
   reindex_helper $i
done
