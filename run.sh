# nvcc ckks23_demo.cu -o ckks23_demo -O3 -lntl -lgmp -ftz=true -prec-div=true
echo "File name is $0"

if [ -n "$1" ]
then
    echo "The \$1 is $1"
else
    echo "\$1 none"
fi

file=$1

if [ "${file##*.}"x = "cu"x ];then
    nvcc -arch=sm_80 $file -o ${file%.*}  -O3 -w -lntl -lgmp
fi

# -cudart shared
# ./ckks23_demo 15
# nvcc test_boot.cu -o boot -O3 -lntl -lgmp -ftz=true -prec-div=true
# ./boot 15