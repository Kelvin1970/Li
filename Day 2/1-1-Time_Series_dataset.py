from pandas import read_csv
import matplotlib.pyplot as plt
dataset = read_csv('dataset/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
print(dataset)
plt.plot(dataset)
plt.show()