#!/usr/bin/env python
import json
import sys

# Get the file (default to timing.txt in pwd)
timingFile = "timing.txt"
timingData = []

# TODO - will extend this to cover output files for each binary
# TODO - Read in multiple files and compare per line using message as the index
if (len(sys.argv) > 1) :
    timingFile = sys.argv[1]

# Read the data as rows of json
with open(timingFile, "r") as timing_file :
    for line in timing_file.readlines() :
        timingData.append(json.loads(line))
        
# Each row is a data line for the timing
for row in timingData :
    print(row["data"]["message"] + "\t" + str(row["data"]["elapsed_time"]/1000.0) + " us")

