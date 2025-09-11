#!/usr/bin/python3

import sys
import os
import blosc
import pickle
import tqdm

try:
    indir = os.path.join(sys.argv[1])
except:
    print('Usage: compress_trackdata.py [INPUT_DIR]')
    exit(1)

trackfiles = [os.path.join(indir, f) for f in os.listdir(indir) if f.endswith('.trackdata')]

assert len(trackfiles) > 0
assert len(os.listdir(indir)) // 2 == len(trackfiles)

print(f"now processing {indir}...")

for infile in tqdm.tqdm(trackfiles):
    with open(infile, 'rb') as f:
        pickled_trackdata = f.read()

    trackdata_c = blosc.compress(pickled_trackdata)

    outfile = infile + '.blosc'
    with open(outfile, 'wb') as f:
        f.write(trackdata_c)