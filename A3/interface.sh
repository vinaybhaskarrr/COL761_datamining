source compile.sh
if [ $1 == "1" ]; then
   python3 part1.py

elif [ $1 == "2" ]; then
    if [ $# -eq 3 ]; then
        if [ $2 == "c" ]; then
            echo $3
            python3 partc.py $3
        else
            echo $3
            python3 partd.py $3
        fi
    else
        echo "Insufficient arguments for part2.py"
    fi
else
    echo "Argument not found"
fi