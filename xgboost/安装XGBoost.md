
gswyhq@gswyhq-PC:~$ docker run -it --name=xgboost -v $PWD/xgboost:/xgboost --workdir="/xgboost" python:3.5.6-stretch /bin/bash
root@67fa994fc45e:/xgboost# git clone --recursive https://github.com/dmlc/xgboost
root@67fa994fc45e:/xgboost# cd xgboost; make -j4
root@67fa994fc45e:/xgboost/python-package# python3 setup.py install

