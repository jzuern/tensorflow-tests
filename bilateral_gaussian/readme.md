# Note on python files of tensorflow installation

The tensorflow installation is in /home/jzuern/tf_installation/tensorflow/

However, the python files being executed when doing a tf graph lie in:

/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops

These files need to be changed in order implement something new




## implementing a custom Op


does not work:
$ bazel build -c opt //tensorflow/core/user_ops:zero_out.so

does work:
$ TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$ g++ -std=c++11 -shared sparse_weighted.cc -o sparse_weighted.so -fPIC -I $TF_INC -I /home/jzuern/tf_installation/tensorflow/ -D_GLIBCXX_USE_CXX11_ABI=0






## TO Do

- übergebe Filter 2 Bilder (to-blur und referenz. Wenn nur 1 Bild angegebn wird, verwende als referenz das bild to-blur)

- Tensorflow-Hashtable anstatt eigene Hash table verwenden



