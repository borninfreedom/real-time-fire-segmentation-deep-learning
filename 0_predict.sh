for i in $(seq 10 20 1000); do
    python scripts/predict.py $i --save_path prediction_$i.png
done