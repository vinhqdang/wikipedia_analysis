import os
import argparse

import matplotlib as plt

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="calculate trust score of wikipedians.")

    parser.add_argument('--input_dir', nargs='?', default='contrib',
                        help='folder contains contribution data')

    return parser.parse_args()

args = parse_args()

qualities = ['ADQ', 'BA', 'A', 'B', 'BD', 'E']

def main ():
    MAX_USER_ID = 100000
    user_weighted_scores = [0] * MAX_USER_ID
    user_raw_scores = [0] * MAX_USER_ID
    user_future_contrib = [0] * MAX_USER_ID
    user_counts = [0] * MAX_USER_ID

    for file_name in os.listdir(args.input_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join (args.input_dir, file_name)
            f = open (file_path)
            quality = f.readline().strip().split(':')[1].upper()
            score = len(qualities) - qualities.index(quality)

            for chunk in iter(lambda: f.readline(), ''):
                user_id, contrib = map (int, chunk.split(':'))
                if user_id < MAX_USER_ID:
                    if user_counts [user_id] < 100:
                        user_weighted_scores [user_id] += score * contrib
                        user_raw_scores [user_id] += contrib
                    else:
                        user_future_contrib [user_id] += contrib
                    user_counts [user_id] += 1
            f.close ()

    max_value = max (max(user_weighted_scores), max(user_raw_scores), max(user_future_contrib))
    user_weighted_scores = [float(x) / max_value for x in user_weighted_scores]
    user_raw_scores = [float(x) / max_value for x in user_raw_scores]
    user_future_contrib = [float(x) / max_value for x in user_future_contrib]

    f = open ("user_contrib.csv", 'w')
    f.write('raw,weight,future\n')
    for i in range(MAX_USER_ID):
        if user_future_contrib[i] > 0:
            f.write (str(user_raw_scores[i]) + ',' + str(user_weighted_scores[i]) + ',' + str(user_future_contrib[i]) + '\n')
    f.close()

if __name__ == '__main__':
    main()