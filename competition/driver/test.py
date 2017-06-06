

import numpy as np
import pandas as pd

if __name__ =="__main__":
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    print df.columns
