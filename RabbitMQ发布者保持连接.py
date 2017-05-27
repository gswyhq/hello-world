#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

# http://stackoverflow.com/questions/35193335/how-to-reconnect-to-rabbitmq

class Publisher:
    EXCHANGE='my_exchange'
    TYPE='topic'
    ROUTING_KEY = 'some_routing_key'

    def __init__(self, host, virtual_host, username, password):
        self._params = pika.connection.ConnectionParameters(
            host=host,
            virtual_host=virtual_host,
            credentials=pika.credentials.PlainCredentials(username, password))
        self._conn = None
        self._channel = None

    def connect(self):
        if not self._conn or self._conn.is_closed:
            self._conn = pika.BlockingConnection(self._params)
            self._channel = self._conn.channel()
            self._channel.exchange_declare(exchange=self.EXCHANGE,
                                           type=self.TYPE)

    def _publish(self, msg):
        self._channel.basic_publish(exchange=self.EXCHANGE,
                                    routing_key=self.ROUTING_KEY,
                                    body=json.dumps(msg).encode())
        logging.debug('message sent: %s', msg)

    def publish(self, msg):
        """Publish msg, reconnecting if necessary."""

        try:
            self._publish(msg)
        except pika.exceptions.ConnectionClosed:
            logging.debug('reconnecting to queue')
            self.connect()
            self._publish(msg)

    def close(self):
        if self._conn and self._conn.is_open:
            logging.debug('closing queue connection')
            self._conn.close()
      

# http://stackoverflow.com/questions/9508246/rabbitmq-pika-and-reconnection-strategy

import logging
import pika
import Queue
import sys
import threading
import time
from functools import partial
from pika.adapters import SelectConnection, BlockingConnection
from pika.exceptions import AMQPConnectionError
from pika.reconnection_strategies import SimpleReconnectionStrategy

log = logging.getLogger(__name__)

DEFAULT_PROPERTIES = pika.BasicProperties(delivery_mode=2)


class Broker(object):

    def __init__(self, parameters, on_channel_open, name='broker'):
        self.parameters = parameters
        self.on_channel_open = on_channel_open
        self.name = name

    def connect(self, forever=False):
        name = self.name
        while True:
            try:
                connection = SelectConnection(
                    self.parameters, self.on_connected)
                log.debug('%s connected', name)
            except Exception:
                if not forever:
                    raise
                log.warning('%s cannot connect', name, exc_info=True)
                time.sleep(10)
                continue

            try:
                connection.ioloop.start()
            finally:
                try:
                    connection.close()
                    connection.ioloop.start() # allow connection to close
                except Exception:
                    pass

            if not forever:
                break

    def on_connected(self, connection):
        connection.channel(self.on_channel_open)


def setup_submitter(channel, data_queue, properties=DEFAULT_PROPERTIES):
    def on_queue_declared(frame):
        # PROBLEM pika does not appear to have a way to detect delivery
        # failure, which means that data could be lost if the connection
        # drops...
        channel.confirm_delivery(on_delivered)
        submit_data()

    def on_delivered(frame):
        if frame.method.NAME in ['Confirm.SelectOk', 'Basic.Ack']:
            log.info('submission confirmed %r', frame)
            # increasing this value seems to cause a higher failure rate
            time.sleep(0)
            submit_data()
        else:
            log.warn('submission failed: %r', frame)
            #data_queue.put(...)

    def submit_data():
        log.info('waiting on data queue')
        data = data_queue.get()
        log.info('got data to submit')
        channel.basic_publish(exchange='',
                    routing_key='sandbox',
                    body=data,
                    properties=properties,
                    mandatory=True)
        log.info('submitted data to broker')

    channel.queue_declare(
        queue='sandbox', durable=True, callback=on_queue_declared)


def blocking_submitter(parameters, data_queue,
        properties=DEFAULT_PROPERTIES):
    while True:
        try:
            connection = BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue='sandbox', durable=True)
        except Exception:
            log.error('connection failure', exc_info=True)
            time.sleep(1)
            continue
        while True:
            log.info('waiting on data queue')
            try:
                data = data_queue.get(timeout=1)
            except Queue.Empty:
                try:
                    connection.process_data_events()
                except AMQPConnectionError:
                    break
                continue
            log.info('got data to submit')
            try:
                channel.basic_publish(exchange='',
                            routing_key='sandbox',
                            body=data,
                            properties=properties,
                            mandatory=True)
            except Exception:
                log.error('submission failed', exc_info=True)
                data_queue.put(data)
                break
            log.info('submitted data to broker')


def setup_receiver(channel, data_queue):
    def process_data(channel, method, properties, body):
        log.info('received data from broker')
        data_queue.put(body)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def on_queue_declared(frame):
        channel.basic_consume(process_data, queue='sandbox')

    channel.queue_declare(
        queue='sandbox', durable=True, callback=on_queue_declared)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'usage: %s RABBITMQ_HOST' % sys.argv[0]
        sys.exit()

    format=('%(asctime)s %(levelname)s %(name)s %(message)s')
    logging.basicConfig(level=logging.DEBUG, format=format)

    host = sys.argv[1]
    log.info('connecting to host: %s', host)
    parameters = pika.ConnectionParameters(host=host, heartbeat=True)
    data_queue = Queue.Queue(0)
    data_queue.put('message') # prime the pump

    # run submitter in a thread

    setup = partial(setup_submitter, data_queue=data_queue)
    broker = Broker(parameters, setup, 'submitter')
    thread = threading.Thread(target=
         partial(broker.connect, forever=True))

    # uncomment these lines to use the blocking variant of the submitter
    #thread = threading.Thread(target=
    #    partial(blocking_submitter, parameters, data_queue))

    thread.daemon = True
    thread.start()

    # run receiver in main thread
    setup = partial(setup_receiver, data_queue=data_queue)
    broker = Broker(parameters, setup, 'receiver')
    broker.connect(forever=True)
    
          
def main():
    pass


if __name__ == "__main__":
    main()
