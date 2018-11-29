from __future__ import print_function
import sys
import argparse

class Ext:
    def __init__(self):
        self.task = 'ext'

    def __proc_bucket(self, bucket):
        # for '-DOCSTART-'
        if len(bucket) == 2:
            print('O')
            print('')
            return None
        for line in bucket[2:-1]:
            print(line)
        print('')
        return None

    def proc(self):
        bucket = []
        while 1:
            try: line = sys.stdin.readline()
            except KeyboardInterrupt: break
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                self.__proc_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0:
            self.__proc_bucket(bucket)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    ext = Ext()
    ext.proc()
