for p in 1 2; do
for n in 1 2 3 4 5 6 7 8 9 10; do
python classifier_vG_split_fine_tune.py --p $p --n $n;
done;
done;