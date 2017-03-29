# separate documents into 6 big text files
# corresponding with 6 quality classes

import os
import sys

in_folder = "text"
out_folder = "separate"
label_file = "ruwikilabel"

file_list = [in_folder+'/'+name for name in os.listdir(in_folder)]

# open (out_folder  + "/" + "fa.txt", "w")
# open (out_folder  + "/" + "ga.txt", "w")
# open (out_folder  + "/" + "sa.txt", "w")
# open (out_folder  + "/" + "i.txt", "w")
# open (out_folder  + "/" + "ii.txt", "w")
# open (out_folder  + "/" + "iii.txt", "w")

labels = []
with open (label_file, "r") as f:
    labels = f.read().splitlines()

if (len(labels) != len(file_list)):
    print ('Error! Number of labels is not as same as number of documents.')
    print ('There are ' + str (len(labels)) + ' labels')
    print ('There are ' + str (len(file_list)) + ' documents')
    sys.exit (1)

index = 0
for file in file_list:
    print ('Processing file number ' + str (index + 1) + ' / ' + str (len(labels)))
    content = open(file,'r').read()
    with open (out_folder + "/" + labels[index] + ".txt", "a") as f:
        f.write (content.replace ("\n"," ") + "\n")
    index += 1