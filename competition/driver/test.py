
import keras
import numpy as np
import pandas as pd

if __name__ =="__main__":
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
    a=np.asarray([1,2,3])
    print df.columns
