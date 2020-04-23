import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import glob

from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data

HIGHLIGHT = "▁@ h igh l ight"

tok_path = get_tokenizer()
sp = SentencepieceTokenizer(tok_path)

def chunk_file(set_name):
    # in_file = 'finished_files/%s.bin' % set_name 
    in_file = os.path.join(finished_files_dir, '%s.bin' % set_name)
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all(chunks_dir, set_name_list):
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in set_name_list:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)

def kobert_tokenizer(sentence):
    sentence = sentence.lower()

    tokens = [token.strip() for token in sp(sentence)]
    return ' '.join(tokens)

def tokenize_stories(stories_dir, tokenized_stories_dir, set_name_list):
    # Maps a whole directory of .story files to a tokenized version
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))

    for set_name in set_name_list:
        set_dir = os.path.join(stories_dir, set_name) # E.g. ./stories/train

        stories = os.listdir(set_dir) # E.g. [tain0.story ...]

        # E.g. ./tokenized_stories_dir/train
        out_dir = os.path.join(os.path.join(tokenized_stories_dir, set_name))
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        for s in stories:
            with open(os.path.join(set_dir, s), "r") as fr, \
                    open(os.path.join(out_dir, s), "w") as fw:
                lines = fr.readlines()
                for line in lines:
                    tokenized = kobert_tokenizer(line)
                    fw.write(tokenized + "\n")

        '''
        Check that the tokenized stories directory contains 
        the same number of files as the original directory
        '''
        num_orig = len(stories)
        num_tokenized = len(os.listdir(out_dir))
        if num_orig != num_tokenized:
            raise Exception("The tokenized stories directory contains %i files, \
                but it should contain the same number as %s (which has %i files). \
                Was there an error during tokenization?" % \
                (num_tokenized, set_dir, num_orig))
        print("Successfully finished tokenizing %s.\n" % (set_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line == "":
            continue # empty line
        elif line.startswith(HIGHLIGHT):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string
    abstract = ' '.join([sent for sent in highlights])

    return article, abstract


def write_to_bin(tokenized_stories_dir, src_name, finished_files_dir, makevocab=False):
    # E.g. ./output/finished_files/train.bin
    out_file = os.path.join(finished_files_dir, src_name + ".bin")
    
    # E.g. ./output/tokenized_stories_dir/train
    set_dir = os.path.join(tokenized_stories_dir, src_name)
    # E.g. [tain0.story ...]
    story_fnames = [name for name in os.listdir(set_dir)]
        
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(story_fnames):
            # Look in the tokenized story dirs to find the .story file corresponding to this url
            if os.path.isfile(os.path.join(set_dir, s)):
                story_file = os.path.join(set_dir, s)
            else:
                print('Error: no data.')

            # Get the strings to write to .bin file
            article, abstract = get_art_abs(story_file)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
        
            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t!=""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':
    stories_dir = "./stories"
    out_dir = './output'

    tokenized_stories_dir = os.path.join(out_dir, "tokenized_stories_dir")
    finished_files_dir = os.path.join(out_dir, "finished_files")
    chunks_dir = os.path.join(finished_files_dir, "chunked")
    
    # Create some new directories
    if not os.path.exists(tokenized_stories_dir): os.makedirs(tokenized_stories_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    set_name_list = ['train', 'val', 'test']

    # Run tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(stories_dir, tokenized_stories_dir, set_name_list)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(tokenized_stories_dir, "train", finished_files_dir, makevocab=True)
    write_to_bin(tokenized_stories_dir, "val", finished_files_dir)
    write_to_bin(tokenized_stories_dir, "test", finished_files_dir)

    # # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all(chunks_dir, set_name_list)