from robin import unet
from keras.models import Model


def test_unet():
    assert unet.unet() == Model
