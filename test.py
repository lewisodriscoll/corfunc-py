from corfunc import *

data = np.loadtxt('example_data/sphere_80.txt',skiprows=1,dtype=np.float32)
q = data[:,0]
iq = data[:,1]

# Create a mathematical curve, s2(q), to fit the data
s2 = fit_data(q,iq,(MINQ,MAXQ))

# Plot the fit curve
# x = np.linspace(q.min(),q.max(),500)
# y = [s2(i) for i in x]
# plt.plot(x,y)
# plt.figure()
# plt.plot(q,iq)
# plt.show()

# Linearly spaced values in q space from 0 to large q
qs = np.arange(0, q[-1]*100, (q[1]-q[0]))
# Value of the best-fit curve at the above values of q
iqs = s2(qs)*qs**2
# Compute the fourier transform of s2(q)
transform = dct(iqs)
xs = np.pi*np.arange(len(qs))/(q[1]-q[0])/len(qs)
# Plot the fit curve and it's fourier transform
# plt.plot(qs,iqs)
# plt.figure()
# plt.plot(xs,transform)
# plt.show()
