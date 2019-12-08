from robin import dataset


def test_dataset():
    assert dataset.split_img_overlay == (list, int, int)
