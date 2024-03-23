module load compiler/gcc/9.1.0
if [ $1 == "1" ] 
then
   python3 part1.py 

elif [ $1 == "2" ] 
then
    if [ $2 =='c' ]; then
        python3 k.py $3 
    else
        python3 part3.py $3
       
    fi
else
 echo "Argument not found"
 
fi