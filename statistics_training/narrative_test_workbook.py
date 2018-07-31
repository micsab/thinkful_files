import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt

# Make the random function consistent and replicable
np.random.seed(1221)

# Make a blank data frame
df = pd.DataFrame()

# Add a column of random numbers between 0 and 1
df['rand'] = np.random.rand(100)
df['rand_sq'] = df['rand'] ** 2
df['rand_shift'] = df['rand'] + 2

# When creating a data frame an index column of counts is created, counting from 0.
# Here we do a few transforms on that index to create some extra columns.
df['counts_sq'] = df.index ** 2
df['counts_sqrt'] = np.sqrt(df.index)

plt.plot(df['rand'])

plt.show()
