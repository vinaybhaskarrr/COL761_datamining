module load compiler/gcc/9.1.0
if [ $1 == "C" ] 
then
   ./compress $2 $3 

elif [ $1 == "D" ] 
then
    ./decompress $2 $3
else
 echo "Argument not found"
 
fi