"""
Loads "cleaned" wikipedia data described at http://mattmahoney.net/dc/textdata
as a list of words. "Cleaned" data uses only the 26 lowercase English
characters.

Data comes in two flavors:
text9: first 10^9 bytes of the English Wikipedia dump on Mar. 3, 2006
text8: first 10^8 bytes of the same dump

Part of this code is based on step 1 of Google's word2vec tutorial available at:
https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
"""

import os
import zipfile
import subprocess
import collections
import numpy as np
from urllib.request import urlretrieve

def maybe_download(rooturl, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(rooturl + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def extract_member(archive, member):
    if not os.path.exists(member):
        with zipfile.ZipFile(archive) as z:
            z.extract(member)
        print("Found and verified " + member)
    else:
        print("Found and verified " + member)

def run_perl(scriptname, arg, stdout=None):
    retcode = subprocess.call(["perl", scriptname, arg], stdout=stdout)
    if not retcode == 0:
        print("Running " + scriptname + " failed!")

def get_text9():
    enwik9 = maybe_download('http://mattmahoney.net/dc/', 'enwik9.zip', 322592222)
    wikifil = maybe_download('http://gist.githubusercontent.com/spitis/176075bc5d4e54b8f03091ef676863bd/raw/dd1b3549842d73690c14a4e8e9a79c7cda1148b5/',
                         'wikifil.pl', 1986)
    extract_member('enwik9.zip','enwik9')

    if os.path.exists('text9'):
        if os.stat('text9').st_size == 713069767:
            print("Found and verified text9")
            return

    with open('text9', "w") as outfile:
        run_perl('wikifil.pl', 'enwik9', outfile)
        print("Found and verified text9")

def clean_text9():
    os.remove('enwik9')
    os.remove('enwik9.zip')
    os.remove('wikifil.pl')

# Read the text8 into a list of strings.
def _load_text8(target_dir = None):
    if target_dir:
        old_dir = os.getcwd()
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        os.chdir(target_dir)

    filename = maybe_download('http://mattmahoney.net/dc/','text8.zip', 31344016)

    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).decode("utf-8").split()

    if target_dir:
        os.chdir(old_dir)

    return data

# Read the text9 into a list of strings.
def _load_text9(target_dir = None):
    if target_dir:
        old_dir = os.getcwd()
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        os.chdir(target_dir)

    if os.path.exists('text9'):
        if os.stat('text9').st_size == 713069767:
            print("Found and verified text9")
    else:
        get_text9()
        clean_text9()

    with open('text9') as f:
        data = f.read().split()

    if target_dir:
        os.chdir(old_dir)

    return data

def load_text8(target_dir = None, vocab_size = 20000):
    text8 = _load_text8(target_dir=target_dir)
    return DataSet(text8, vocab_size)

def load_text9(target_dir = None, vocab_size = 50000):
    text9 = _load_text9(target_dir=target_dir)
    return DataSet(text9, vocab_size)

class DataSet(object):
    def __init__(self, list_of_words, vocab_size=20000):
        count = [['UNK', -1]]
        count.extend(collections.Counter(list_of_words).most_common(vocab_size - 1))
        dictionary = {}
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = []
        unk_count = 0
        for word in list_of_words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        self._dataidx = 0
        self._datalen = len(data)
        self.vocab_size = vocab_size
        self.data = data
        self.count = count
        self.word_to_idx = dictionary
        self.idx_to_word = reverse_dictionary

    def generate_batch(self, batch_size, skip_window=2):
        samples_per_target = 2 * skip_window
        assert batch_size % samples_per_target == 0

        targets = np.ndarray(shape=(batch_size), dtype=np.int32)
        contexts = np.ndarray(shape=(batch_size), dtype=np.int32)

        buffer = collections.deque(maxlen=samples_per_target + 1)
        cursor = self._dataidx

        for _ in range(samples_per_target + 1):
            buffer.append(self.data[cursor])
            cursor = (cursor + 1) % self._datalen

        # for each target word in this batch
        for i in range(batch_size // samples_per_target):
            start = i * samples_per_target

            for j in range(samples_per_target):
                if j >= skip_window:
                    k = j+1
                else:
                    k = j
                targets[start + j] = buffer[skip_window]
                contexts[start + j] = buffer[k]
            buffer.append(self.data[cursor])
            cursor = (cursor + 1) % self._datalen

        self._dataidx = (self._dataidx + (batch_size // samples_per_target)) % self._datalen

        return targets, contexts
