for l in 2 3; do
for n in 2 3 4 5 6 7 8 9 10; do
python classifier_vG_fine_tune.py --l $l --n $n;
done;
done;