trap 'exit' INT

python w5.py -test t0 > task1_results/t0.txt 2>&1

python w5.py -test t1 > task1_results/t1.txt 2>&1

python w5.py -test t2 > task1_results/t2.txt 2>&1

python w5.py -test t3 > task1_results/t3.txt 2>&1

python w5.py -test t4 > task1_results/t4.txt 2>&1

python w5.py -test t5 > task1_results/t5.txt 2>&1

python w5.py -test t6 > task1_results/t6.txt 2>&1
