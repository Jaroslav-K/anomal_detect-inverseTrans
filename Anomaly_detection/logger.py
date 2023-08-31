import tensorflow as tf
from datetime import datetime

class Logger:

    def __init__(self, output_stream):
        self.output_stream = output_stream
        self.log(f'\n{datetime.now().strftime("%d.%m.%Y_%H:%M:%S")}')

    def log(self, msg):
        tf.print(msg, summarize=-1, output_stream=self.output_stream)