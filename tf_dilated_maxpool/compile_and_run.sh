TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
nvcc --compiler-bindir /usr/bin/gcc-4.8 -std=c++11 -c -o dilated_maxpooling_gpu.cu.o dilated_maxpooling_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared -o dilated_maxpooling_gpu.so dilated_maxpooling_gpu.cc dilated_maxpooling_gpu.cu.o -I $TF_INC -I /home/jzuern/tensorflow/ -fPIC -lcudart -D_GLIBCXX_USE_CXX11_ABI=0
python test_customOp_cuda.py
