import sys
import os

def is_there_alphabet(sentence):
    for i in range(65, 91):
        if (chr(i) in sentence) or (chr(i).lower() in sentence):
            return True
    return False

def is_there_math_symbol(sentence):
    math_symbol = ['+', '-', '×', '÷', '=', '|', '`', 'ù', '~', 'æ' ,'ª']
    for symbol in math_symbol:
        if symbol in sentence:
            return True
    return False

# convert dataset from tsv to story (format : sentence1 \n\n@highlight\n\n sentence2)
def tsv2stories(tsv_dir, stories_dir):
    """
    :param tsv_dir: path of tsv directory
    :param stories_dir: path of output directory
    """

    tsv_fnames = [name for name in os.listdir(tsv_dir)]

    idx = 0
    for fname in tsv_fnames:
        with open(os.path.join(tsv_dir, fname), "r") as fin:
            for i, line in enumerate(fin):
                if i == 0: continue # first line is 'chunk'
                chunk = line.strip().split("\t")[-1]
                if is_there_alphabet(chunk): continue
                if is_there_math_symbol(chunk): continue
                chunk = chunk[1:].strip() # Ignore first character, e.g. (1)
                if chunk == '': continue # empty chunk

                with open(os.path.join(stories_dir, 'test%d.story'%idx), "w") as fout:
                    print(chunk, file=fout)
                    print("\n@highlight\n", file = fout)
                    print(chunk, file=fout)
                idx += 1

    print('Source data has been converted from "{0}" to "{1}".'.format(tsv_dir, stories_dir))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python tsv_to_stories.py <tsv_dir>")
        sys.exit()

    tsv_dir = sys.argv[1]
    stories_dir = "./stories/test"

    # Create some new directories
    if not os.path.exists(stories_dir): os.makedirs(stories_dir)

    tsv2stories(tsv_dir, stories_dir)