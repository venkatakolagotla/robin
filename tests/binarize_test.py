from __future__ import print_function

from robin import binarize
import shutil


def test_main(data_path):
    print(data_path)
    output = binarize.main(data_path)
    assert(type(output) == list)
    # Remove the directory created
    shutil.rmtree('output')
