#!/bin/bash

CURRENTDIR=$(pwd)
CLASSNAMES=($(cd original_samples && echo */))
SAMPLEAMOUNT=1500
TRAIN_SAMPLE=1000
VALIDATAION_SAMPLE=200
REINDEX_SCRIPT="reindex.sh"
AUGMENT_SCRIPT="augment_samples.py"

# Before script
rm -rf train validation test
mkdir -p train validation train
for i in "${CLASSNAMES[@]}"
do
    classname=${i%/}
    mkdir -p train/$i validation/$i test/$i
    no_of_elements=$(ls original_samples/$i | wc -l)
    python "$AUGMENT_SCRIPT" $CURRENTDIR/original_samples/$i \
        $CURRENTDIR/test/$i \
        $(($SAMPLEAMOUNT/$no_of_elements+1))

    # Reindex
    cp $REINDEX_SCRIPT test
    cd test && bash ../"$REINDEX_SCRIPT" $classname; cd $CURRENTDIR

    # Distribute training samples.
    for (( j = 0; j < $(echo $(($TRAIN_SAMPLE/100))); j++ )); do
        block=$(printf "%02d" $j)
        mv test/$classname/"$classname""$block"* train/$classname
    done

    # Distribute validation samples.
    for (( j = $(echo $(($TRAIN_SAMPLE/100))); j < $(echo $(($TRAIN_SAMPLE/100+$VALIDATAION_SAMPLE/100))); j++ )); do
        block=$(printf "%02d" $j)
        mv test/$classname/"$classname""$block"* validation/$classname
    done
done

