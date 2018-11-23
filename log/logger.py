#!/usr/bin/python
# coding:utf8

import os
import logging, threading
import traceback
import logging.handlers


class Logger(object):
    uid = ''
    show_source_location = True

    def __init__(self):
        #debug log
        self.log = logging.getLogger("semLog")
        self.log.setLevel(logging.DEBUG)

        outer = logging.StreamHandler()
        outer.setLevel(logging.DEBUG)
        outer.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(threadName)s-%(message)s'))
        self.log.addHandler(outer)

        file_outer = logging.handlers.RotatingFileHandler("log/yhb.log", mode='a', maxBytes=1024 * 1024 * 100,
                                                          backupCount=10000, encoding="utf-8")
        file_outer.setLevel(logging.DEBUG)
        file_outer.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(threadName)s-%(message)s'))
        self.log.addHandler(file_outer)

        #io log
        self.iolog = logging.getLogger("semIo")
        self.iolog.setLevel(logging.INFO)
        file_outer = logging.handlers.RotatingFileHandler("log/io.log", mode='a', maxBytes=1024 * 1024 * 100,
                                                          backupCount=10000, encoding="utf-8")
        file_outer.setLevel(logging.INFO)
        file_outer.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(threadName)s-%(message)s'))
        self.iolog.addHandler(file_outer)

        # error log
        self.errlog = logging.getLogger("semErr")
        self.errlog.setLevel(logging.ERROR)
        file_outer = logging.handlers.RotatingFileHandler("log/error.log", mode='a', maxBytes=1024 * 1024 * 100,
                                                          backupCount=10000, encoding="utf-8")
        file_outer.setLevel(logging.ERROR)
        file_outer.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(threadName)s-%(message)s'))
        self.errlog.addHandler(file_outer)

        # 图灵接口的日志
        self.tulingiolog = logging.getLogger("semtulingIo")
        self.tulingiolog.setLevel(logging.INFO)
        file_outer = logging.handlers.RotatingFileHandler("log/tuling.log", mode='a', maxBytes=1024 * 1024 * 100,
                                                          backupCount=10000, encoding="utf-8")
        file_outer.setLevel(logging.INFO)
        file_outer.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(threadName)s-%(message)s'))
        self.tulingiolog.addHandler(file_outer)

    # Formats the message as needed and calls the correct logging method
    # to actually handle it
    def _raw_log(self, logfn, message, exc_info, no_uid=True):
        cname = ''
        loc = ''
        fn = ''
        tb = traceback.extract_stack()
        if len(tb) > 2:
            if self.show_source_location:
                loc = '(%s:%d):' % (os.path.basename(tb[-3][0]), tb[-3][1])
            fn = tb[-3][2]
            if fn != '<module>':
                if self.__class__.__name__ != Logger.__name__:
                    fn = self.__class__.__name__ + '.' + fn
                fn += '()'

        # logfn(self.uid + loc + cname + fn + ': ' + str(message), exc_info=exc_info)
        if no_uid:
            logfn(u"{}{}{}: {}".format(loc, cname, fn, message), exc_info=exc_info)
        else:
            logfn(u"{}{}{}{}: {}".format(self.uid, loc, cname, fn, message), exc_info=exc_info)

    def info(self, message, exc_info=False, no_uid=True):
        """
        Log a info-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(self.iolog.info, message, exc_info, no_uid=no_uid)

    def tulinginfo(self, message, exc_info=False, no_uid=True):
        """
        Log a info-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(self.tulingiolog.info, message, exc_info, no_uid=no_uid)

    def debug(self, message, exc_info=False, no_uid=True):
        """
        Log a debug-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(self.log.debug, message, exc_info, no_uid=no_uid)

    def warning(self, message, exc_info=False, no_uid=True):
        """
        Log a warning-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(self.errlog.warning, message, exc_info, no_uid=no_uid)

    def error(self, message, exc_info=False):
        """
        Log an error-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(self.errlog.error, message, exc_info)
        self._raw_log(self.iolog.error, message, exc_info)

    def exception(self, message, exc_info=False):
        """
        Log an error-level message. If exc_info is True, if an exception
        was caught, show the exception information (message and stack trace).
        """
        self._raw_log(self.errlog.exception, message, exc_info)
        self._raw_log(self.iolog.exception, message, exc_info)

    def set_uid(self, uid):
        # self.uid = uid
        # 将用户id设置为线程名
        threading.current_thread().setName(uid)

logger = Logger()

def set_logger_uid(uid):
    logger.set_uid(uid)
