module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/matplotlib/3.0.2/gnu
module load compiler/gcc/11.2.0
chmod 777 *
python3 one.py $1 FSG
python3 one.py $1 GSPAN
python3 one.py $1 GASTON
python3 q1.py $2