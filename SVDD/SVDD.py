from random import choice, random
import os
from numpy import array, zeros, ndarray, where, isnan, intersect1d, ones
import matplotlib.pyplot as plt

from kernel import LinearKernel, RBFKernel, PolyKernel

# 设置随机种子
from numpy.random import seed; seed(0)
from random import seed; seed(0)
import torch
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import shutil

class SupportVectorDataDescription:
    def __init__(self,
            C: float = 0.5,
            kernel: str = 'linear',
            maxIterations: int = 10000,
            tol: float = 1e-3,
            d: int = 2,
            γ: float = 1.,
            r: float = 1.,
            ):
        assert C>0, 'Penalty parameter C should be greater than 0'
        assert type(maxIterations)==int and maxIterations>0, 'The maximum number of iterations, maxIterations, should be a positive integer'
        assert tol>0, 'The convergence accuracy tol should be greater than 0'
        assert type(d)==int and d>=1, 'The exponent d of a polynomial kernel function should be a positive integer not less than 1'
        assert γ>0, 'Parameters of Gaussian kernel function and polynomial kernel function γ Should be greater than 0'
        self.C = C
        self.kernel = kernel.lower()
        self.maxIterations = maxIterations
        self.tol = tol
        self.d = d
        self.γ = γ
        self.r = r
        self.M = None
        self.a_ = None
        self.aTa = None
        self.R = None
        self.α_ = None
        self.supportVectors__ = None
        self.αSV_ = None
        self.losses_ = None
        """Select kernel function"""
        if self.kernel=='linear':
            self.kernelFunction = LinearKernel()
            print('Using linear kernel function')
        elif self.kernel=='poly':
            self.kernelFunction = PolyKernel(d=self.d, γ=self.γ, r=self.r)
            print('Using polynomial kernel function')
        elif self.kernel=='rbf':
            self.kernelFunction = RBFKernel(γ=self.γ)
            print('Using Gaussian kernel function')
        else:
            raise ValueError(f"Undefined kernel function'{kernel}'")

    def fit(self, X__: ndarray):
        assert type(X__)==ndarray and X__.ndim==2, 'Input training sample matrix X__ Should be a 2D darray'
        C = self.C
        N, self.M = X__.shape
        assert C>1/N, f'Penalty parameter C should be greater than 1/N = {1/N}'
        K__ = self.kernelFunction(X__, X__)
        diagK_ = K__.diagonal()
        α_ = ones(N)/N
        R = 1.
        self.losses_ = losses_ = []
        aTa = α_ @ K__ @ α_
        loss = aTa - α_ @ diagK_
        for t in range(1, self.maxIterations + 1):
            indexSV_ = where(α_>0)[0]
            indexNonBound_ = where((0<α_) & (α_<C))[0]
            losses_.append(loss)
            g_ = α_[indexSV_] @ K__[indexSV_, :]
            D_ = abs(aTa + diagK_ - 2*g_)**0.5
            violateKKT_ = abs(D_ - R)
            violateKKT_[(α_==0) & (D_<=R)] = 0.
            violateKKT_[(0<α_) & (α_<C) & (D_==R)] = 0.
            violateKKT_[(α_==C) & (D_>=R)] = 0.
            if violateKKT_.max()<self.tol:
                print(f'The {t}th SMO iteration, the maximum degree of violation of KKT conditions reaches convergence accuracy {self.tol}, stop the iteration!')
                break
            indexViolateKKT_ = where(violateKKT_>0)[0]
            indexNonBoundViolateKKT_ = intersect1d(indexViolateKKT_, indexNonBound_)
            if random()<0.85:
                if len(indexNonBoundViolateKKT_)>0:
                    i = indexNonBoundViolateKKT_[violateKKT_[indexNonBoundViolateKKT_].argmax()]
                else:
                    i = violateKKT_.argmax()
            else:
                i = choice(indexViolateKKT_)
            j = choice(indexViolateKKT_)
            while (X__[i]==X__[j]).all():
                j = choice(range(N))
            print(f'The {t} SMO iteration, select i= = {i}, j = {j}')
            ζ = α_[i] + α_[j]
            αiOld, αjOld = α_[i], α_[j]
            Kii = K__[i, i]
            Kjj = K__[j, j]
            Kij = K__[i, j]
            η = Kii + Kjj - 2*Kij
            L, H = max(0, ζ - C), min(C, ζ)
            αj = αjOld + (g_[i] - g_[j] + 0.5*(Kjj - Kii))/η
            if αj>H:
                αj = H
            elif αj<L:
                αj = L
            else:
                pass
            αi = ζ - αj
            if αi>C:
                αi = C
            elif αi<0:
                αi = 0
            else:
                pass
            α_[j], α_[i] = αj, αi
            vi = g_[i] - αiOld*Kii - αjOld*Kij
            vj = g_[j] - αiOld*Kij - αjOld*Kjj
            aTaOld = aTa
            aTa += ( (αi**2 - αiOld**2)*Kii
                   + (αj**2 - αjOld**2)*Kjj
                   + 2*(αi*αj - αiOld*αjOld)*Kij
                   + 2*vi*(αi - αiOld)
                   + 2*vj*(αj - αjOld)
                    )
            loss += ( (aTa - aTaOld)
                    -diagK_[i]*(αi - αiOld)
                    -diagK_[j]*(αj - αjOld)
                    )
            if 0<α_[i]<C:
                R = abs(aTa + diagK_[i] - 2*(g_[i] + (αi - αiOld)*Kii + (αj - αjOld)*Kij))**0.5
            elif 0<α_[j]<C:
                R = abs(aTa + diagK_[j] - 2*(g_[j] + (αi - αiOld)*Kij + (αj - αjOld)*Kjj))**0.5
            else:
                Ri = abs(aTa + diagK_[i] - 2*(g_[i] + (αi - αiOld)*Kii + (αj - αjOld)*Kij))**0.5
                Rj = abs(aTa + diagK_[j] - 2*(g_[j] + (αi - αiOld)*Kij + (αj - αjOld)*Kjj))**0.5
                R = 0.5*(Ri + Rj)
            if isnan(R):
                raise ValueError('The radius R value of the metasphere is nan, troubleshooting error!')
        else:
            print(f'Reached maximum number of iterations{self.maxIterations}!')

        indexSV_ = where(α_>0)[0]
        self.αSV_ = α_[indexSV_]
        self.α_ = α_
        self.aTa = self.α_ @ K__ @ self.α_
        self.supportVectors__ = X__[indexSV_]
        indexNonBound_ = where((0<α_) & (α_<C))[0]
        if len(indexNonBound_)>0:
            R_ = [(self.aTa + diagK_[k] - 2*self.αSV_ @ K__[indexSV_, k])**0.5
                  for k in indexNonBound_]
            self.R = sum(R_)/len(R_)
        else:
            self.R = R
            print('There is no satisfying α， Take the radius R of the hypersphere obtained from the last iteration')
        if self.kernel=='linear':
            self.a_ = self.αSV_@self.supportVectors__
        return self

    def predict(self, X__: ndarray) -> ndarray:
        assert type(X__)==ndarray and X__.ndim==2, 'Input Test Sample Matrix X__ Should be a 2D darray'
        assert X__.shape[1]==self.M, f'The dimension of the input test sample should be equal to the dimension of the training sample{self.M}'
        y_ = self.decisionFunction(X__)<=self.R  # 正常 True/异常 False
        return y_

    def decisionFunction(self, X__) -> ndarray:
        assert X__.ndim==2, 'Input sample matrix X__ Should be a 2D darray'
        assert X__.shape[1]==self.M, f'The input sample dimension should be equal to the training sample dimension{self.M}'
        d_ = zeros(len(X__))
        for n, x_ in enumerate(X__):
            d2 = self.aTa\
                 + self.kernelFunction(x_, x_) \
                 - 2*self.αSV_ @ self.kernelFunction(x_, self.supportVectors__)  # 样本x_到超球体球心的距离平方
            d_[n] = abs(d2)**0.5
        return d_

    def abnormalRate(self, X__: ndarray)-> float:

        pred = self.predict(X__)
        Nss = sum(pred)
        Tts = len(X__)
        abnormal_rate = (Tts - Nss)/Tts
        return abnormal_rate, pred

    def train2D(self, X__):
        if self.M!=2:
            print('When the dimension of the input feature vector is 2, the drawing can be called')
            return
        import numpy as np
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.set_title(f'{self.kernel.capitalize()} kernel function; $C$ = {self.C:g}')  # 图标题：核函数名称、惩罚参数C
        h = []
        h += ax.plot(X__[:, 0], X__[:, 1], 'bo')
        # h += ax.plot(self.supportVectors__[:, 0],
        #              self.supportVectors__[:, 1],
        #              'ok',
        #              markersize=15,
        #              color = 'black',
        #              markerfacecolor='none')
        x0range, x1range = X__[:, 0].ptp(), X__[:, 1].ptp()
        x0_ = np.linspace(X__[:, 0].min() - 0.25*x0range, X__[:, 0].max() + 0.25*x0range, 200)
        x1_ = np.linspace(X__[:, 1].min() - 0.25*x1range, X__[:, 1].max() + 0.25*x1range, 200)
        grid__ = [[self.decisionFunction(array([[x0, x1]]))[0]
                   for x0 in x0_]
                   for x1 in x1_]
        ax.contour(x0_, x1_, grid__,
                   levels=[self.R],
                   linestyles='--',
                   colors='k')
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        # ax.legend(h,
        #           ['Samples', 'Support vectors']
        #          )
        ax.set_aspect('equal')

    def test2D(self, X__):
        if self.M != 2:
            print('When the dimension of the input feature vector is 2, the drawing can be called')
            return
        import numpy as np
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.set_title(f'{self.kernel.capitalize()} kernel function; $C$ = {self.C:g}')
        h = []
        h += ax.plot(X__[:, 0], X__[:, 1], 'ro')
        h += ax.plot(self.supportVectors__[:, 0],
                     self.supportVectors__[:, 1],
                     'ok',
                     markersize=0,
                     markerfacecolor='none')
        x0range, x1range = X__[:, 0].ptp(), X__[:, 1].ptp()
        x0_ = np.linspace(X__[:, 0].min() - 0.25 * x0range, X__[:, 0].max() + 0.25 * x0range, 200)
        x1_ = np.linspace(X__[:, 1].min() - 0.25 * x1range, X__[:, 1].max() + 0.25 * x1range, 200)
        grid__ = [[self.decisionFunction(array([[x0, x1]]))[0]
                   for x0 in x0_]
                  for x1 in x1_]
        ax.contour(x0_, x1_, grid__,
                   levels=[self.R],
                   linestyles='--',
                   colors='k')
        ax.set_xlabel('$x_{0}$')
        ax.set_ylabel('$x_{1}$')
        # ax.legend(h,
        #           ['Samples', 'Support vectors']
        #           )
        ax.set_aspect('equal')

    def plotLoss(self):
        import numpy as np
        losses_ = np.array(self.losses_)
        fig = plt.figure(figsize=[6, 6])
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(losses_) + 1), losses_, 'r-')
        ax.plot(range(1, len(losses_)), losses_[:-1] - losses_[1:], 'b-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Minimized objective function')
        ax.legend(['Function value', 'Decrement of function value'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = './AE/bloodmnist'
encoder = torch.load(path + '/bloodmnist.pth')
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
    ])
for i in range(7):
    dataset = ImageFolder("./data/bloodmnist/BPRT-GP/%d" % (i), transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    imgs_o, _ = next(iter(dataloader))
    imgs_o = Variable(imgs_o).to(device)
    o_e = encoder(imgs_o)
    o_ls = o_e.cpu().view(-1, 4 * 4 * 32).detach().numpy()

    pca = PCA(n_components=2, random_state=42)
    pca = pca.fit(o_ls)
    o = pca.transform(o_ls)

    model_path = './model/bloodmnist/'
    img_path = './image/bloodmnist/'
    model = torch.load(model_path + 'model_%d.pth' % i)
    model.test2D(0)
    plt.savefig(img_path + '%d.png' % (i), dpi=800)
    plt.show()
    if model.kernel == 'linear':
        print(f'Hypersphere centroid vector a_ = {model.a_}')
    print(f'Hypersphere radius R = {model.R}')
    tr_ab, tr_pred = model.abnormalRate(o)
    print(f'Abnormal rate of training set：{tr_ab:.5f}')
    index = [i for i, x in enumerate(tr_pred.tolist()) if x == False]
    print(index)
    # Remove Outlier
    save_path = './data/bloodmnist/IBGAN/SVDD/SVDD_%d/' % (i)
    os.makedirs(save_path, exist_ok=True)
    t = 0
    for j in range(len(index)):
        shutil.move(dataset.imgs[index[j]][0], save_path + '%d.png' % (t))
        t = t + 1


