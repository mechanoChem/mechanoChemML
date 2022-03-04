import os
import pip

def get_cuda_version():
    def get_version(output_info):
        return output_info.split(' ')[-1].strip()[1:].split('.')

    cuda_version = None
    output_info = os.popen("nvcc --version").read()
    if output_info.find('release') >= 0:
        cuda_version = get_version(output_info)
    output_info = os.popen("/usr/local/cuda/bin/./nvcc --version").read()
    if output_info.find('release') >= 0:
        cuda_version = get_version(output_info)
    return cuda_version

# install tensorflow that is compatible with the detected cuda version
cuda_version = get_cuda_version()
if cuda_version is not None:
    if cuda_version[0] == '10':
        pip.main(["install", "tensorflow>=2.2,<2.4"])
    elif cuda_version[0] == '11':
        pip.main(["install", "tensorflow>=2.4"])
# install tensorflow without cuda support as cuda is not detected
else:
    pip.main(["install", "tensorflow>=2.2"])

import tensorflow as tf
# install tensorflow_probability
def force_install(package, versions=None):

    """install one package with versions """

    if versions is not None:
        pip.main(['install', package+'=='+versions])
    else:
        pip.main(['install', package])

def check_and_install_tfp(package, versions=None):

    """make sure tfp is compatiable with tf """

    try:
        __import__(package)
    except ImportError:
        force_install(package, versions)
    import tensorflow_probability as tfp
    tfp_version=tfp.__version__.split(".")
    if tfp_version[0:2] != versions.split(".")[0:2]:
        force_install(package, versions)

tf_version=tf.__version__.split(".")
if tf_version[0] == '2' and tf_version[1] == '8': check_and_install_tfp('tensorflow_probability', '0.16.0')
if tf_version[0] == '2' and tf_version[1] == '7': check_and_install_tfp('tensorflow_probability', '0.15.0')
if tf_version[0] == '2' and tf_version[1] == '6': check_and_install_tfp('tensorflow_probability', '0.14.0')
if tf_version[0] == '2' and tf_version[1] == '5': check_and_install_tfp('tensorflow_probability', '0.13.0')
if tf_version[0] == '2' and tf_version[1] == '4': check_and_install_tfp('tensorflow_probability', '0.12.0')
if tf_version[0] == '2' and tf_version[1] == '3': check_and_install_tfp('tensorflow_probability', '0.11.0')
if tf_version[0] == '2' and tf_version[1] == '2': check_and_install_tfp('tensorflow_probability', '0.10.0')
