#!/bin/bash

cd `pwd`
rm -rf ProvinceCityAreaClean.c ProvinceCityAreaClean.cpython-36m-x86_64-linux-gnu.so build 
python3.6 setup.py build_ext --inplace

mv ProvinceCityAreaClean.cpython-36m-x86_64-linux-gnu.so libPCACleanTest.so

