# split or concatenate text file
# in order to by pass the limitation of 100MB of github

import os
import sys

if len(sys.argv) <= 2:
    print ('Error! Missing parameter')
    print ('-s text_file for split')
    print ('-j file_base for concatenate')
    sys.exit (1)

if (sys.argv[1] == "-s"):
    file_name = sys.argv[2]
    file_cnt = 1
    MAX_LINE = 1000 # maximum lines per file
    with open(file_name, 'r') as f:
        line_cnt = 0
        sub_filename = file_name + "_" + str(file_cnt)
        for line in f:            
            if line_cnt < MAX_LINE:
                with open(sub_filename, "a") as f2:
                    f2.write (line)
                line_cnt += 1
            if line_cnt == MAX_LINE:
                line_cnt = 0
                file_cnt += 1
                sub_filename = file_name + "_" + str(file_cnt)
                # remove previous subfilename if any
                try:
                    os.remove (sub_filename)
                except Exception, e:
                    pass
elif (sys.argv[1] == "-j"):
    file_name = sys.argv[2]
    # remove previous big file
    try:
        os.remove(file_name)
    except Exception, e:
        pass
    file_cnt = 1
    while True:
        sub_filename = file_name + "_" + str (file_cnt)
        if os.path.isfile(sub_filename):
            sub_content = []
            with open (subfilename, 'r') as f1:
                for line in f1:
                    with open (file_name, 'a') as f2:
                        f2.write (line)
            file_cnt += 1
        else:
            break
else:
    print ('Error! Missing parameter')
    print ('-s text_file for split')
    print ('-j file_1 file_2 ... for concatenate')
    sys.exit (1)