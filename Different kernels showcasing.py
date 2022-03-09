import numpy as np
import math as m
import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler

def w(v): #The the condion for rectrangular function
    if np.abs(v) < 1/2:
        return 1
    else:
        return 0


def epanechnikov(a, b, c):
    lt = 0
    u = 0
    for i in range(len(a)):
        u = (m.sqrt((a[i][0]-b[0])**2 + (a[i][1]-b[1])**2)/c)
        if u <= 1:
            lt += 0.75*(1-u**2)
    return float(lt)


def triangular(a, b, c):
    lt = 0
    u = 0
    for i in range(len(a)):
        u = (m.sqrt((a[i][0]-b[0])**2 + (a[i][1]-b[1])**2)/c)
        if u <= 1:
            lt += (1-np.abs(u))
    return float(lt)


def rectangularwindow(a, b, c):
    co = 0
    u = 0
    for i in range(len(a)):
        u = (m.sqrt((a[i][0]-b[0])**2 + (a[i][1]-b[1])**2)/c)
        if w(u) == 1:
            co += 1
    t = co / (c * c * len(a))
    return float(t)


def sigmoid(a, b, c):
    lt = 0
    u = 0
    for i in range(len(a)):
        u = (m.sqrt((a[i][0]-b[0])**2 + (a[i][1]-b[1])**2)/c)
        lt += (2/m.pi)*1/(m.exp(-(u))+m.exp(u))
    return float(lt)


def silverman(a, b, c):
    lt = 0
    u = 0
    for i in range(len(a)):
        u = (m.sqrt((a[i][0]-b[0])**2 + (a[i][1]-b[1])**2)/c)
        lt += m.exp(-(np.abs(u)/m.sqrt(2)))*m.sin((np.abs(u)/m.sqrt(2))+(m.pi/4))
    return float(lt)


def gaussian(a, b, c):
    lt = 0
    t = 0
    for i in range(len(a)):
        lt += (1/((2*m.pi)*c*c))*m.exp(-0.5*(((a[i][0]-b[0])**2 + (a[i][1]-b[1])**2)/c**2))
    t = float(lt)
    return t


mu = np.array([0, 0])
cov = np.array([[1, 0], [0, 1]])
# center = [[1, 1], [-1, 1], [1, -1]]
# ga, ol = make_blobs(n_samples=10000, centers=center, cluster_std = 0.4, random_state=0)
# ga = StandardScaler().fit_transform(ga)
ga = np.random.multivariate_normal(mu, cov, 10000)
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
fig = plt.figure(figsize=(14, 9))
ax = plt.axes(projection='3d')
f, qx = plt.subplots(figsize=(7, 7))
qx.scatter(ga[:, 0], ga[:, 1], marker='o', color='red', s=4, alpha=0.3)
plt.title('10000 samples randomly drawn from a 2D Gaussian distribution')
plt.ylabel('x2')
plt.xlabel('x1')
ftext = 'Inital Data'
plt.figtext(.15, .85, ftext, fontsize=11, ha='left')
plt.ylim([-4, 4])
plt.xlim([-4, 4])
Z = []
u = 0
ty = int(input('Enter which type of Kernel do you want to use? \n1. Rectrangular kernel \n2. Triangular kernel\n3. Gaussian kernel\n4. Sigmoid kernel\n5. Silverman kernel\n6. Epanechnikov kernel\nEnter the the index number:'))
h = float(input("Enter the Width of the kernel:"))
if(ty == 1):
    for i, j in zip(X.ravel(), Y.ravel()):
        u += 1
        print(f'The epoch is:{u}')
        Z.append(rectangularwindow(ga, np.array([[i], [j]]), h))
elif(ty == 2):
    for i, j in zip(X.ravel(), Y.ravel()):
        u += 1
        print(f'The epoch is:{u}')
        Z.append(triangular(ga, np.array([[i], [j]]), h))
        break
elif(ty == 3):
    for i, j in zip(X.ravel(), Y.ravel()):
        u += 1
        print(f'The epoch is:{u}')
        Z.append(gaussian(ga, np.array([[i], [j]]), h))
elif(ty == 4):
    for i, j in zip(X.ravel(), Y.ravel()):
        u += 1
        print(f'The epoch is:{u}')
        Z.append(sigmoid(ga, np.array([[i], [j]]), h))
elif(ty == 5):
    for i, j in zip(X.ravel(), Y.ravel()):
        u += 1
        print(f'The epoch is:{u}')
        Z.append(silverman(ga, np.array([[i], [j]]), h))
elif(ty == 6):
    for i, j in zip(X.ravel(), Y.ravel()):
        u += 1
        print(f'The epoch is:{u}')
        Z.append(Epanechnikov(ga, np.array([[i], [j]]), h))
else:
    print("Enter a valid input")
Z = np.asarray(Z).reshape(100, 100)
ax.set_title(f"Hypercube kernel with window width {h}")
ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
plt.show()
