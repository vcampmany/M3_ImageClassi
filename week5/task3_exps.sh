trap 'exit' INT

mkdir -p task3_results/zoom

python w5.py -test t2 -train_folder train_400 --dataug > task3_results/zoom/t2_train400.txt 2>&1
python w5.py -test t2 -train_folder train_200 --dataug > task3_results/zoom/t2_train200.txt 2>&1
python w5.py -test t2 -train_folder train_100 --dataug > task3_results/zoom/t2_train100.txt 2>&1
python w5.py -test t2 -train_folder train_50 --dataug > task3_results/zoom/t2_train50.txt 2>&1
python w5.py -test t2 -train_folder train_24 --dataug > task3_results/zoom/t2_train24.txt 2>&1
python w5.py -test t2 -train_folder train_8 --dataug > task3_results/zoom/t2_train8.txt 2>&1