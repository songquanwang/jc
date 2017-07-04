import keras
import numpy as np
import pandas as pd
import theano
import theano.tensor as T

if __name__ == "__main__":
    # dates = pd.date_range('20130101', periods=6)
    # df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    # y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    # a = np.asarray([1, 2, 3])
    # print df.columns

    k = T.iscalar('k')
    A = T.vector('A')

    def fun(result,A):
        return result * A

    outputs, updates = theano.scan(fun, non_sequences=A, outputs_info=T.ones_like(A), n_steps=k)
    result = outputs[-1]
    fn_Ak = theano.function([A, k], result, updates=updates)
    r =fn_Ak(range(10), 2)
    print r