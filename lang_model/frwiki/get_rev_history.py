"""
parse JSON file and get history of editing
"""

import json
from pprint import pprint
import urllib2
import re
import os
import datetime
from dateutil import parser

# import distance
# import Levenshtein
import editdistance

import argparse

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Retrieve historical contribution of Wikipedia users.")

    parser.add_argument('--start_index', type = int, nargs='?', default=0,
                        help='starting file index.')

    return parser.parse_args()

args = parse_args()

lines = []
non_decimal = re.compile(r'[^\d.]+')

with open("frwiki.labelings.9k.json", "r") as json_file:
    lines = json_file.readlines()

label_file = "frwikilabel"

# try:
#     os.remove(label_file)
# except OSError:
#     pass

missing_lines = []

for i in range(len(lines)):
# for i in [1]: # for debug
    if i < args.start_index:
        continue
    print ('Processing line number ' + str (i+1) + ' / ' + str(len(lines)))
    # if (os.path.isfile ("text/" + str (i+1))):
    #     continue
    json_object = json.loads (lines[i])
    project = json_object["project"]
    title = json_object["page_title"]
    timestamp = json_object["timestamp"]
    quality = json_object["wp10"]
    # print title
    # url_string = "https://fr.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&titles=" + title.replace (" ","%20")
    # action=query&format=json&prop=revisions&meta=&titles=Ferdinand+William+Hutchison&rvprop=ids%7Ctimestamp&rvlimit=500
    
    # get the list of all revisions of the page
    
    # rvstart is the starting timestamp we want to collect revision
    # set it before the original timestamp so we can collect the correct one
    rvstart = timestamp[0:4] + "-" + timestamp[4:6] + "-" + timestamp [6:8] + "T" + timestamp[8:10] + ":" + timestamp[10:12] + ":" + timestamp[12:14] + "Z"
    datetime_object = parser.parse(rvstart)
    # go further 1 day to makesure that we cover the correct revision
    datetime_object = datetime_object + datetime.timedelta(hours=1)
    rvstart = datetime_object.strftime ("%Y-%m-%dT%H:%M:%SZ")
    # print (rvstart)
    # raw_input()

    url_string = "https://fr.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=ids%7Ctimestamp&rvlimit=500&format=json&titles=" + title.replace (" ","%20").encode("utf8") + "&rvstart=" + rvstart
    # print ('url: ' + url_string)
    # raw_input()
    response = urllib2.urlopen(url_string).read()
    json_response = json.loads (response)

    page_id = json_response["query"]["pages"].keys()[0]
    try:
        revisions = json_response["query"]["pages"][page_id]["revisions"]
    except Exception, e:
        print ('Missing at line ' + str(i + 1))
        print (url_string)
        missing_lines.append(i+1)
        print ('So far ' + str (len (missing_lines)) + " missing pages")
        continue    

    # print (revisions[1])

    # look for revision whose timestamp is equal to timestamp in original JSON file
    found_revision = False
    last_content = ''
    user_ids = []
    user_contrib = []
    for j in reversed (range(len(revisions))):
        print ('Processing revision ' + str(len(revisions) - j) + '/' + str(len(revisions)) + ' of file: ' + str(i+1))
        rev_timestamp = revisions[j]["timestamp"]
        rev_id = revisions[j]["revid"]
        rev_timestamp = non_decimal.sub('', rev_timestamp)
        # print (rev_timestamp)
        # the quality class is assigned at timestamp, therefore we collect the content of the last revision
        # before the timestamp
        # if (rev_timestamp < timestamp):
        #     # print ('Found!')
        #     # print (rev_timestamp)
        #     # print (timestamp)
        #     # print (rev_id)
        #     found_revision = True
        #     break

    # if found_revision == False:
    #     print ("Cannot find the correct revision")
    #     print ("Title: " +  title)
    #     print ("earliest timestamp: " + str(rev_timestamp))
    #     print ("original timestamp: " + str(timestamp))
    #     missing_lines.append(i+1)
    #     print ('So far ' + str(len(missing_lines)) + " missing")
    #     print (url_string)
    #     # raw_input("Press anykey to skip and continue in next line")
    #     continue

    # got rev_id, now get the data
        url_string = "https://fr.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content%7Cuserid&format=json&revids=" + str(rev_id)
        response = urllib2.urlopen(url_string).read ()
        json_response = json.loads (response)


        page_id = json_response["query"]["pages"].keys()[0]
        try:
            content = json_response["query"]["pages"][page_id]["revisions"][0]["*"]
            user_id = json_response["query"]["pages"][page_id]["revisions"][0]["userid"]
        except Exception, e:
            print ('Cannot get text')
            print (url_string)
            # print (json_response)
            print (e)
            missing_lines.append(i+1)
            continue

        # print user_id
        print ('Calculating new contribution')
        new_contrib = editdistance.eval (content, last_content)
        # new_contrib = abs (len(content) - len(last_content))
        last_content = content
        print ('User ' + str(user_id) + ' contribute: ' + str (new_contrib))
        user_ids.append(user_id)
        user_contrib.append(new_contrib)

    # write user contribution to file
    file_name = "contrib/" + str (i+1) + '.txt'
    f = open (file_name, "w")
    f.write("quality:" + quality + '\n')
    for k in range (len (user_ids)):
        f.write (str(user_ids[k]) + ':' + str(user_contrib[k]) + '\n')
    f.close()

    # write content to file
    # file_name = "text/" + str(i+1)
    # with open (file_name, "w") as f:
    #     f.write(content.encode('utf8'))
    # with open (label_file, "a") as f:
    #     f.write (str (i+1) + '\t' + quality.encode('utf8') + "\n")

print ('Total missing page = ' + str (len(missing_lines)) + " over " + str(len(lines)) + " pages.")
print (missing_lines)