# a new file
import math
import matplotlib.pyplot as plt

a =  5.2917721067 * 10 ** (-11.0)
alpha = 0.0072973525664

H = []
He = []
alpha_beta = []
beta_val = []

for beta in range(1, 10000):
    a_b = ((alpha/(beta/10000.0)) ** 2.0)
    cross_H = 8 * math.pi * (a ** 2.0) * (a_b) * (2.42 - 1.93 * (a_b))

    cross_He = 8 * math.pi * (a ** 2.0) * (a_b) * (2.81 - 8.78 * (a_b))

    H.append(10000 * cross_H)
    He.append(10000 * cross_He)
    beta_val.append(beta)
    print(beta)
    print(cross_H)
    print(cross_He)

    alpha_beta.append(a_b)
    print(a_b)
#print H, He
print(a, alpha)
#print beta_val
#print alpha_beta

fig = plt.figure
ax1 = plt.subplot(111)
plt.plot(alpha_beta, H)
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.gca().invert_xaxis()
#plt.xlim((0.12, 0.0001))
#plt.ylim(((10 ** (-19)), (2 * 10 ** (-15))))
ax2 = ax1.twinx()
plt.plot(alpha_beta, He, 'g')
ax2.set_xscale('log')
ax2.set_yscale('log')
#plt.gca().invert_xaxis()
#plt.gca().invert_yaxis()
#plt.xlim((0.12, 0.0001))
#plt.ylim(((10 ** (-20)), (2 * 10 ** (-16))))
plt.show()

input(" Press Enter to finish ... ")

