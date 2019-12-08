from robin import binarize
import shutil


def test_binarize():
    output = binarize.main(input="test_data/input_imgs")
    assert(type(output) == list)
    # Remove the directory created
    shutil.rmtree('output')
