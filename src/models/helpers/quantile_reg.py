import numpy as np



def pinball_loss_2(y_true, y_pred, **kwargs):
    print('y_true',type(y_true))
    print('y_pred',type(y_pred))
    return 1


def pinball_loss(y_true, y_pred, quantile, **kwargs):
    if not all(isinstance(i,np.ndarray) for i in [y_true, y_pred]):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    diff = np.subtract(y_true, y_pred)
    pinball_loss = np.multiply(diff, np.where(diff > 0, quantile, (1-quantile)))
    # print(sum(abs(pinball_loss))/len(y_true))
    return sum(abs(pinball_loss))/len(y_true)


def partial(func, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs[:2], **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc
