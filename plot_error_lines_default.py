#!/usr/bin/env python

"""
Plot an x,y line graph...
"""

import sys
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot error lines given a log file.')
parser.add_argument('file', metavar='f', type=str, nargs='?',
                   help='a log file to process')
parser.add_argument('label', metavar='l', type=str, nargs='?',
                   help='the label lines of interest start with')
parser.add_argument('xcol', metavar='x', type=int, nargs='?', default=4,
                   help='the column number containing the xs')
parser.add_argument('ycol', metavar='y', type=int, nargs='?', default=5,
                   help='the column number containing the ys')

args = parser.parse_args()

def main():
    xrecon = []
    xcross = []
    xentr = []
    ys = []
    for l in open(args.file):
        l = l.strip()
        if l.startswith(args.label):
            toks = l.split()
            print toks
            if 'LRecon' in l:
                xrecon.append(toks[args.ycol - 1])
            elif 'LCross' in l:
                xcross.append(toks[args.ycol - 1])
            elif 'LEntr' in l:
                xentr.append(toks[args.ycol - 1])
            ys.append(int(toks[args.xcol - 1]))
    xrecon = [float(x) for x in xrecon]
    xcross = [float(x) for x in xcross]
    xentr = [float(x) for x in xentr]
    ys = sorted(list(set(ys)))
    plt.subplot(311)
    plt.title('Reconstruction loss')
    plt.axis('off')
    plt.plot(ys,xrecon)
    plt.subplot(312)
    # plt.title('Cross entropy loss')
    # plt.axis('off')
    plt.plot(ys,xcross,label='discriminator')
    # plt.subplot(313)
    # plt.title('Entropy loss')
    plt.axis('off')
    plt.plot(ys,xentr,label='generator')
    plt.legend(loc="lower left")
    plt.show()

if __name__ == '__main__':
    main()
