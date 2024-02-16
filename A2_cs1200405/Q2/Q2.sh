module load compiler/gcc/11.2.0
chmod 777 *
g++ q2.cpp -o q2
./q2 $1
./gSpan-64 -f gspanin.txt -s 0.1 -o -i
g++ check.cpp -o check
./check
g++ final.cpp -o final
./final $2
