module load compiler/gcc/9.1.0
g++ compression.cpp -o compress -O3
g++ decompression.cpp -o decompress -O3