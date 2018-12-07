from __future__ import print_function
import os
import sys
import argparse
import copy
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a_path', type=str, help='a path', required=True)
    parser.add_argument('--b_path', type=str, help='a path', required=True)

    args = parser.parse_args()

    a_data = []
    bucket = []
    for line in open(args.a_path):
        line = line.strip()
        if not line and len(bucket) >= 1:
            entry = copy.deepcopy(bucket)
            a_data.append(entry)
            bucket = []
        else:
            bucket.append(line)

    b_data = []
    bucket = []
    for line in open(args.b_path):
        line = line.strip()
        if not line and len(bucket) >= 1:
            entry = copy.deepcopy(bucket)
            b_data.append(entry)
            bucket = []
        else: bucket.append(line)

    '''
    print(len(a_data))
    print(len(b_data))
    for idx in range(len(a_data)):
        print(len(a_data[idx]), len(b_data[idx]))
    sys.exit(0)
    '''

    idx = 0
    for idx, _ in enumerate(a_data):
        if len(a_data[idx]) != len(b_data[idx]):
            sys.stderr.write('merge error!')
            break
            print('[before]')
            for a_line in a_data[idx-1]:
                print(a_line)
            print('')
            for b_line in b_data[idx-1]:
                print(b_line)
            print('')
            print('[inconsistent] %d vs %d' % (len(a_data[idx]), len(b_data[idx])))
            for a_line in a_data[idx]:
                print(a_line)
            print('')
            for b_line in b_data[idx]:
                print(b_line)
            print('')
            print('[next]')
            for a_line in a_data[idx+1]:
                print(a_line)
            print('')
            for b_line in b_data[idx+1]:
                print(b_line)
            print('==========================================')

        for a_line, b_line in zip(a_data[idx], b_data[idx]):
            a_tokens = a_line.split()
            if a_tokens[3] == 'X': continue
            if b_line == 'X': b_line = 'O'
            print(a_line, b_line)
        print('')
        
        
     
