from numpy import  ndarray, exp, zeros

class LinearKernel:

    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        if X__.ndim==1 and 1<=Z__.ndim<=2:
            return Z__ @ X__
        elif 1<=X__.ndim<=2 and Z__.ndim==1:
            return X__ @ Z__
        elif X__.ndim==2 and Z__.ndim==2:
            K__ = zeros([len(X__), len(Z__)])
            for n, x_ in enumerate(X__):
                K__[n] = Z__ @ x_
            return K__
        else:
            raise ValueError('Input quantity X__ Z__ Expected ndarray, its attribute ndim should be 1 or 2 ')

class PolyKernel:
    def __init__(self,
            d: int = 2,
            r: float = 1.,
            γ: float = 1.,
            ):
        assert type(d)==int and d>=1, 'The exponent d of a polynomial kernel function should be a positive integer, K(x_, z_) = (γ * x_ @ z_ + r)**d'
        assert γ>0, 'Hyperparameters of polynomial kernel functions γ Should be a positive number, K(x_, z_) = (γ * x_ @ z_ + r)**d'
        self.d = d
        self.r = r
        self.γ = γ
        self.linearKernel = LinearKernel()

    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        K__ = self.linearKernel(X__, Z__)
        K__ = (self.γ*K__ + self.r)**self.d
        return K__

class RBFKernel:

    def __init__(self, γ: float = 1.):
        assert γ>0, 'Hyperparameters of RBF kernel functions γ Expected positive number'
        self.γ = γ

    def __call__(self, X__: ndarray, Z__: ndarray) -> ndarray:
        if ((X__.ndim==1 and Z__.ndim==2) or
            (X__.ndim==2 and Z__.ndim==1)):
            return exp(-self.γ * ((X__ - Z__)**2).sum(axis=1))
        elif X__.ndim==1 and Z__.ndim==1:
            return exp(-self.γ * ((X__ - Z__)**2).sum())
        elif X__.ndim==2 and Z__.ndim==2:
            D2__ = (X__**2).sum(axis=1, keepdims=True) + (Z__**2).sum(axis=1) - 2*X__ @ Z__.T
            return exp(-self.γ * D2__)
        else:
            raise ValueError('Input quantity X__ Z__ Expected ndarray, whose attribute ndim should be 1 or 2')
