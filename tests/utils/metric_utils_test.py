from robin.utils import metric_utils


def test_dice_coef(y_true, y_pred):
    # dice coefficient test
    dice_coef = metric_utils.dice_coef(y_true, y_pred)
    assert dice_coef.dtype == float


def test_dice_coef_loss(y_true, y_pred):
    # dice loss test
    dice_loss = metric_utils.dice_coef_loss(y_true, y_pred)
    assert dice_loss.dtype == float


def test_jaccard_coef(y_true, y_pred):
    # jaccard coefficient test
    jaccard_coef = metric_utils.jacard_coef(y_true, y_pred)
    assert jaccard_coef.dtype == float


def test_jaccard_coef_loss(y_true, y_pred):
    # jacccard loss test
    jaccard_loss = metric_utils.jacard_coef_loss(y_true, y_pred)
    assert jaccard_loss.dtype == float
