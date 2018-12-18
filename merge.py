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

    assert(len(a_data) == len(b_data))

    m_data = []
    for idx, _ in enumerate(a_data):
        if len(a_data[idx]) != len(b_data[idx]):
            sys.stderr.write('merge error!')
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
            sys.exit(1)

        entry = []
        for a_line, b_line in zip(a_data[idx], b_data[idx]):
            a_tokens = a_line.split()
            assert(len(a_tokens) == 4)
            a_word = a_tokens[0].replace('##','')
            a_pos  = a_tokens[1]
            a_chunk = a_tokens[2]
            a_label = a_tokens[3]
            b_label = b_line
            if b_label == 'X': b_label = 'O'
            tp = (a_word, a_pos, a_chunk, a_label, b_label)
            entry.append(tp)
        m_data.append(entry)
        
    for idx, entry in enumerate(m_data):
        size = len(entry)
        seq = 0
        while seq != size:
            tp = entry[seq]
            word, pos, chunk, label, predict = tp
            rng = 1
            if seq + rng >= size:
                print(word, pos, chunk, label, predict)
                break
            n_seq = seq + rng
            n_tp = entry[n_seq]
            n_word, n_pos, n_chunk, n_label, n_predict = n_tp
            org_word = word
            while n_label == 'X':
                org_word += n_word
                rng += 1
                n_seq = seq + rng
                if seq + rng >= size: break
                n_tp = entry[n_seq]
                n_word, n_pos, n_chunk, n_label, n_predict = n_tp
            print(org_word, pos, chunk, label, predict)
            seq += rng
        print('')
