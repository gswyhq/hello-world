
import os
import datetime
import pyinotify
import logging
class MyEventHandler(pyinotify.ProcessEvent):
    logging.basicConfig(level=logging.INFO,filename='/var/log/monitor.log')
    #自定义写入那个文件，可以自己修改
    logging.info("Starting monitor...")
     
    def process_IN_ACCESS(self, event):
        print "ACCESS event:", event.pathname
    logging.info("ACCESS event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
    def process_IN_ATTRIB(self, event):
        print "ATTRIB event:", event.pathname
    logging.info("IN_ATTRIB event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
   def process_IN_CLOSE_NOWRITE(self, event):
        print "CLOSE_NOWRITE event:", event.pathname
        logging.info("CLOSE_NOWRITE event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
   def process_IN_CLOSE_WRITE(self, event):
        print "CLOSE_WRITE event:", event.pathname
    logging.info("CLOSE_WRITE event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
   def process_IN_CREATE(self, event):
        print "CREATE event:", event.pathname
    logging.info("CREATE event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
   def process_IN_DELETE(self, event):
        print "DELETE event:", event.pathname
    logging.info("DELETE event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
   def process_IN_MODIFY(self, event):
        print "MODIFY event:", event.pathname
    logging.info("MODIFY event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
   def process_IN_OPEN(self, event):
        print "OPEN event:", event.pathname
    logging.info("OPEN event : %s  %s" % (os.path.join(event.path,event.name),datetime.datetime.now()))
     
     
def main():
    # watch manager
    wm = pyinotify.WatchManager()
    wm.add_watch('/tmp', pyinotify.ALL_EVENTS, rec=True)
    #/tmp是可以自己修改的监控的目录
    # event handler
    eh = MyEventHandler()
 
    # notifier
    notifier = pyinotify.Notifier(wm, eh)
    notifier.loop()
 
if __name__ == '__main__':
    main()
