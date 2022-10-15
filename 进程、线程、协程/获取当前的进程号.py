#!/usr/bin/python3
# coding: utf-8

import os
from logger.logger import logger

def pid():
    pid = os.getpid()
    logger.info("当前进程号： {}".format(pid))
    return pid

def main():
    pass


if __name__ == '__main__':
    main()