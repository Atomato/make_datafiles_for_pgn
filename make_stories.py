import sys
import os
import pandas as pd

# convert dataset from xlsx to story (format : sentence1 \n\n@highlight\n\n sentence2)
def xlsx2stories(xlsx_path, stories_dir):
    """
    :param xlsx_path: path of dataset file
    :param stories_dir: path of output dir
    """

    # 파일을 읽고 줄로 분리
    file = pd.read_excel(xlsx_path)
    pairs = [[file['train_x'][i], file['train_y'][i]] for i in range(len(file))]

    pairs_len = len(pairs) # 페어 개수

    train_num = (pairs_len // 20) * 19 # train 페어 개수
    pairs_train = pairs[:train_num] 
    pairs_val = pairs[train_num:] 

    # story 파일로 변환
    for idx, pair in enumerate(pairs_train):
        with open(os.path.join(stories_dir, 'train%d.story'%idx), 'w') as fout:   
            print(pair[0], file = fout)
            print("\n@highlight\n", file = fout)
            print(pair[1], file = fout)

    for idx, pair in enumerate(pairs_val):
        with open(os.path.join(stories_dir, 'val%d.story'%idx), 'w') as fout:   
            print(pair[0], file = fout)
            print("\n@highlight\n", file = fout)
            print(pair[1], file = fout)

    print('Source data has been converted from "{0}" to "{1}".'.format(xlsx_path, stories_dir))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python make_stories.py <xlsx_path> <stories_dir>")
        sys.exit()

    xlsx_path = sys.argv[1]
    stories_dir = sys.argv[2]

    # Create some new directories
    if not os.path.exists(stories_dir): os.makedirs(stories_dir)

    xlsx2stories(xlsx_path, stories_dir)

