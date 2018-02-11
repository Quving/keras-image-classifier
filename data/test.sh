classname1="foo"
classname2="bar"
TRAIN_SAMPLE=1000
VALIDATAION_SAMPLE=200

for (( j = 0; j < $(echo $(($TRAIN_SAMPLE/10))); j++ )); do
    block=$(printf "%02d" $j)
    # echo "$classname1""$block"
done

# Distribute validation samples.
for (( j = $(echo $(($TRAIN_SAMPLE/10))); j < $(echo $(($TRAIN_SAMPLE/10+$VALIDATAION_SAMPLE/10))); j++ )); do
    block=$(printf "%02d" $j)
    # echo "$classname2""$block"
done
$(echo $(($TRAIN_SAMPLE/10)));
$(echo $(($TRAIN_SAMPLE/10+$VALIDATAION_SAMPLE/10)));
