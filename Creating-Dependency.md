# Supported region

Currently all the six lambda regions AMI specific dependencies are already present for tensorflow 0.9.0. We need to use the below script only in case of new regions or different version of tensorflow.

# Adding New dependency for different region.

This [link](http://docs.aws.amazon.com/lambda/latest/dg/current-supported-versions.html) shows the list of AMI used for different regions of AWS LAMBDA. To create deployment package we need to create dependency files with the corresponding AMI. Spin up the corresponding AMI and execute the follwing commands to create the deployment package. The scrip finally boots up a webserver from which you can down the dependency zip file.
 
Create a pull request in [this repository](https://github.com/anandanand84/aws-lambda-tensorflow-dependencies) to add the dependency for different version or region.  

You need to add your application specific file to this folder and zip the files in the folder to create the deployment package.

# Deployment Package dependency creation script

```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
sudo yum -y update
sudo yum -y upgrade
sudo yum -y group install "Development Tools"
sudo yum -y install Cython --enablerepo=epel
sudo yum -y install python27-devel python27-pip gcc

virtualenv --always-copy ~/tensorflow_env
source ~/tensorflow_env/bin/activate

python2.7 —-version
which python2.7
which pip
pip install --upgrade pip
pip --version
pip install --upgrade ${TF_BINARY_URL}
deactivate

cd ~/tensorflow_env/lib/python2.7/site-packages
touch google/__init__.py
zip -r9v ~/lambda-tensorflow-dependency-ap-southeast-2.zip . --exclude \*.pyc
cd ~/tensorflow_env/lib64/python2.7/site-packages
zip -r9v ~/lambda-tensorflow-dependency-ap-southeast-2.zip . --exclude \*.pyc
cd ~
python -m SimpleHTTPServer 8000

```
