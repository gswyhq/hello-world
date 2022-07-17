#!/bin/bash

cd `pwd`
rm -rf *.class /tmp/*

javac -encoding UTF-8 PCAClean.java Demo.java
java -Djava.library.path='.' Demo

jar cvfm DTest.jar manf *.class libPCACleanTest.so addtopost.pkl
# java -jar DTest.jar

