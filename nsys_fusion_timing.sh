#!/bin/bash

problem_size_m=${problem_size_m:-1024}
problem_size_n=${problem_size_n:-1024}
problem_size_k=${problem_size_k:-1024}
kernel=${kernel:-gemm}

# Get the passed parameter values if any.
while [ $# -gt 0 ]; do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi

  shift
done

# Calculate flops.
((flops = $problem_size_m * $problem_size_n * $problem_size_k * 2))

# Profile
nsys profile --force-overwrite true -o gpu_ ./a.out $problem_size_m $problem_size_n $problem_size_k 10 2> dump_.txt

# Get average execution time of `turing`.
matmul=$(nsys stats -q --force-overwrite true --format csv --report gpukernsum gpu_.qdrep | (awk -v kernel="gemm" '$0~kernel') | (awk -F',' '{print $4}'))
matAdd=$(nsys stats -q --force-overwrite true --format csv --report gpukernsum gpu_.qdrep | (awk -v kernel="matAdd" '$0~kernel') | (awk -F',' '{print $4}'))
relu=$(nsys stats -q --force-overwrite true --format csv --report gpukernsum gpu_.qdrep | (awk -v kernel="pwRelu" '$0~kernel') | (awk -F',' '{print $4}'))

rm -f gpu_.qdrep
rm -f gpu_.sqlite
rm -f dump_.txt

# Check if perf is reported by `nsys`.
if [ -z "$matmul" ]
then
    matmul=0
    #echo -e "\e[31merror:\e[0m" "matmul execTime was not given by nsys."
fi

if [ -z "$matAdd" ]
then
    #echo -e "\e[31merror:\e[0m" "pw addition execTime was not given by nsys."
    matAdd=0
fi

if [ -z "$relu" ]
then
    #echo -e "\e[31merror:\e[0m" "relu addition execTime was not given by nsys."
    relu=0
fi

# Calculate performance.
>&2 printf '%.6f matmulTFLOPs, ' $(echo "(($flops / ($matmul)) * 1000000000) / 1000000000000" | bc -l)
>&2 printf '%.6f fusedTFLOPs\n' $(echo "(($flops / ($matAdd+$matmul+$relu)) * 1000000000) / 1000000000000" | bc -l)
