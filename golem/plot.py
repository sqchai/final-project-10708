from numpy import genfromtxt
import matplotlib.pyplot as plt

# w_real = genfromtxt('W_true.csv', delimiter=',')
# print(w_real.max(), w_real.min())
# plt.imshow(w_real , cmap = 'Blues' )
# plt.axis('off')
# plt.savefig('real', dpi=100)

w_pred = genfromtxt('W_est.csv', delimiter=',')
print(w_pred.max(), w_pred.min())
plt.clf()
plt.imshow(w_pred , cmap = 'Blues' )
plt.axis('off')
plt.savefig('est', dpi=100)