trap 'exit' INT

mkdir -p task4_results/dataug
mkdir -p task4_results/no_dataug

python w5.py -test t2 -train_folder train --dataug --dropout > task4_results/dataug/t2_trainALL.txt 2>&1
python w5.py -test t2 -train_folder train_400 --dataug --dropout > task4_results/dataug/t2_train400.txt 2>&1
python w5.py -test t2 -train_folder train_200 --dataug --dropout > task4_results/dataug/t2_train200.txt 2>&1
python w5.py -test t2 -train_folder train_100 --dataug --dropout > task4_results/dataug/t2_train100.txt 2>&1
python w5.py -test t2 -train_folder train_50 --dataug --dropout > task4_results/dataug/t2_train50.txt 2>&1
python w5.py -test t2 -train_folder train_24 --dataug --dropout > task4_results/dataug/t2_train24.txt 2>&1
python w5.py -test t2 -train_folder train_8 --dataug --dropout > task4_results/dataug/t2_train8.txt 2>&1

python w5.py -test t2 -train_folder train --dropout > task4_results/no_dataug/t2_trainALL.txt 2>&1
python w5.py -test t2 -train_folder train_400 --dropout > task4_results/no_dataug/t2_train400.txt 2>&1
python w5.py -test t2 -train_folder train_200 --dropout > task4_results/no_dataug/t2_train200.txt 2>&1
python w5.py -test t2 -train_folder train_100 --dropout > task4_results/no_dataug/t2_train100.txt 2>&1
python w5.py -test t2 -train_folder train_50 --dropout > task4_results/no_dataug/t2_train50.txt 2>&1
python w5.py -test t2 -train_folder train_24 --dropout > task4_results/no_dataug/t2_train24.txt 2>&1
python w5.py -test t2 -train_folder train_8 --dropout > task4_results/no_dataug/t2_train8.txt 2>&1