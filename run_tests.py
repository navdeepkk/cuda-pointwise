# Usage: python run_tests.py

import subprocess
import re

problem_size_m = ["1024", "1280", "1536", "1792", "2048","2304", "2560", "2816", "3072", "3328", "3584", "4096", "3840", "4352", "4608", "4864", "5120", "5376", "5632", "5888", "6144", "6400", "6656", "6912", "7168", "7424", "7680", "7936", "8192", "8448", "8704", "8960", "9216", "9472", "9728", "9984", "10240", "10496", "10752", "11008", "11264", "11520", "11776", "12032", "12288", "12544", "12800", "13056", "13312", "13568", "13824", "14080", "14336", "14592", "14848", "15104", "15360", "15616", "15872", "16128", "16384"]
problem_size_n = ["1024", "1280", "1536", "1792", "2048","2304", "2560", "2816", "3072", "3328", "3584", "4096", "3840", "4352", "4608", "4864", "5120", "5376", "5632", "5888", "6144", "6400", "6656", "6912", "7168", "7424", "7680", "7936", "8192", "8448", "8704", "8960", "9216", "9472", "9728", "9984", "10240", "10496", "10752", "11008", "11264", "11520", "11776", "12032", "12288", "12544", "12800", "13056", "13312", "13568", "13824", "14080", "14336", "14592", "14848", "15104", "15360", "15616", "15872", "16128", "16384"]
problem_size_k = ["1024", "1280", "1536", "1792", "2048","2304", "2560", "2816", "3072", "3328", "3584", "4096", "3840", "4352", "4608", "4864", "5120", "5376", "5632", "5888", "6144", "6400", "6656", "6912", "7168", "7424", "7680", "7936", "8192", "8448", "8704", "8960", "9216", "9472", "9728", "9984", "10240", "10496", "10752", "11008", "11264", "11520", "11776", "12032", "12288", "12544", "12800", "13056", "13312", "13568", "13824", "14080", "14336", "14592", "14848", "15104", "15360", "15616", "15872", "16128", "16384"]

#subprocess.run(["/usr/local/cuda-11.2/bin/nvcc", "-gencode",  "arch=compute_86,code=sm_86", "-lcublas", "gemm_cst_add.cu"])
#print("cst_add")
#print("problem_size_m, problem_size_n, problem_size_k, GFLOPs")
#
#for pm in problem_size_m:
#    for pn in problem_size_n:
#        for pk in problem_size_k:
#            if int(pm) == int(pn) and int(pn) == int(pk):
#                result = subprocess.run(["./nsys_fusion_timing.sh", "--problem_size_m", pm, "--problem_size_n", pn, "--problem_size_k", pk, "--kernel", "gemm"], capture_output=True)
#                res = result.stderr
#                tidy = re.findall('\d*\.?\d+', res.decode('utf-8'))                                                              
#                print(pm,", ",pn,", ",pk,", ",tidy)
#
#subprocess.run(["/usr/local/cuda-11.2/bin/nvcc", "-gencode",  "arch=compute_86,code=sm_86", "-lcublas", "gemm_mat_add.cu"])
#print("mat_add")
#
#for pm in problem_size_m:
#    for pn in problem_size_n:
#        for pk in problem_size_k:
#            if int(pm) == int(pn) and int(pn) == int(pk):
#                result = subprocess.run(["./nsys_fusion_timing.sh", "--problem_size_m", pm, "--problem_size_n", pn, "--problem_size_k", pk, "--kernel", "gemm"], capture_output=True)
#                res = result.stderr
#                tidy = re.findall('\d*\.?\d+', res.decode('utf-8'))                                                              
#                print(pm,", ",pn,", ",pk,", ",tidy)
#
#subprocess.run(["/usr/local/cuda-11.2/bin/nvcc", "-gencode",  "arch=compute_86,code=sm_86", "-lcublas", "gemm_mat_add_relu.cu"])
#print("mat_add_relu")
#
#for pm in problem_size_m:
#    for pn in problem_size_n:
#        for pk in problem_size_k:
#            if int(pm) == int(pn) and int(pn) == int(pk):
#                result = subprocess.run(["./nsys_fusion_timing.sh", "--problem_size_m", pm, "--problem_size_n", pn, "--problem_size_k", pk, "--kernel", "gemm"], capture_output=True)
#                res = result.stderr
#                tidy = re.findall('\d*\.?\d+', res.decode('utf-8'))                                                              
#                print(pm,", ",pn,", ",pk,", ",tidy)

subprocess.run(["/usr/local/cuda-11.2/bin/nvcc", "-gencode",  "arch=compute_86,code=sm_86", "-lcublas", "relu_gemm.cu"])
print("relu_gemm")

for pm in problem_size_m:
    for pn in problem_size_n:
        for pk in problem_size_k:
            if int(pm) == int(pn) and int(pn) == int(pk):
                result = subprocess.run(["./nsys_fusion_timing.sh", "--problem_size_m", pm, "--problem_size_n", pn, "--problem_size_k", pk, "--kernel", "gemm"], capture_output=True)
                res = result.stderr
                tidy = re.findall('\d*\.?\d+', res.decode('utf-8'))                                                              
                print(pm,", ",pn,", ",pk,", ",tidy)
