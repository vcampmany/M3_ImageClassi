trap 'exit' INT

mkdir -p task2_results

python w5.py -test t2 -train_folder train_400 > task2_results/t2_train400.txt 2>&1
python w5.py -test t2 -train_folder train_200 > task2_results/t2_train200.txt 2>&1
python w5.py -test t2 -train_folder train_100 > task2_results/t2_train100.txt 2>&1
python w5.py -test t2 -train_folder train_50 > task2_results/t2_train50.txt 2>&1
python w5.py -test t2 -train_folder train_24 > task2_results/t2_train24.txt 2>&1