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
    zcross = []
    ycross = []
    zentr = []
    yentr = []
    ys = []
    for l in open(args.file):
        l = l.strip()
        if l.startswith(args.label):
            toks = l.split()
            print toks
            if 'LRecon' in l:
                xrecon.append(toks[args.ycol - 1])
            elif 'Z-Cross' in l:
                zcross.append(toks[args.ycol - 1])
            elif 'Y-Cross' in l:
                ycross.append(toks[args.ycol - 1])
            elif 'Z-Entr' in l:
                zentr.append(toks[args.ycol - 1])
            elif 'Y-Entr' in l:
                yentr.append(toks[args.ycol - 1])
            ys.append(int(toks[args.xcol - 1]))
    ys = sorted(list(set(ys)))
    xrecon = [float(x) for x in xrecon]
    zcross = [float(x) for x in zcross]
    ycross = [float(x) for x in ycross]
    zentr = [float(x) for x in zentr]
    yentr = [float(x) for x in yentr]
    plt.subplot(511)
    plt.title('Reconstruction loss')
    plt.plot(ys,xrecon)
    plt.axis('off')
    plt.subplot(512)
    # plt.title('Z Cross entropy loss')
    plt.plot(ys,zcross,label='Z_discriminator')
    plt.axis('off')
    # plt.subplot(513)
    # plt.title('Y Cross entropy loss')
    plt.plot(ys,zentr,label='Z_generator')
    plt.legend(loc="lower left")
    plt.axis('off')
    plt.subplot(514)
    # plt.title('Z Entropy loss')
    plt.plot(ys,ycross,label='Y_discriminator')
    plt.axis('off')
    # plt.subplot(515)
    # plt.title('Y Entropy loss')
    plt.plot(ys,yentr,label='Y_generator')
    plt.axis('off')
    plt.legend(loc="lower left")
    plt.show()

if __name__ == '__main__':
    main()
