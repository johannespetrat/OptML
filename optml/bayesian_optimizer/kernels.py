import numpy as np

from sklearn.gaussian_process.kernels import Hyperparameter

from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import Kernel as sk_Kernel
from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from sklearn.gaussian_process.kernels import StationaryKernelMixin
from sklearn.gaussian_process.kernels import WhiteKernel as sk_WhiteKernel

class Kernel(sk_Kernel):
    """
    Base class for skopt.gaussian_process kernels.
    Supports computation of the gradient of the kernel with respect to X
    """
    def __add__(self, b):
        if not isinstance(b, Kernel):
            raise Exception, "Need to some with another kernel"
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def __pow__(self, b):
        return Exponentiation(self, b)

    def gradient_x(self, x, X_train):
        """
        Computes gradient of K(x, X_train) with respect to x
        Parameters
        ----------
        x: array-like, shape=(n_features,)
            A single test point.
        Y: array-like, shape=(n_samples, n_features)
            Training data used to fit the gaussian process.
        Returns
        -------
        gradient_x: array-like, shape=(n_samples, n_features)
            Gradient of K(x, X_train) with respect to x.
        """
        raise NotImplementedError


class Sum(Kernel, sk_Sum):

    def gradient_x(self, x, X_train):
        return (
            self.k1.gradient_x(x, X_train) +
            self.k2.gradient_x(x, X_train)
        )

class WhiteKernel(Kernel, sk_WhiteKernel):

    def gradient_x(self, x, X_train):
        return np.zeros_like(X_train)


class HammingKernel(StationaryKernelMixin, NormalizedKernelMixin,
                    Kernel):
    """
    The HammingKernel is used to handle categorical inputs.
    ``K(x_1, x_2) = exp(\sum_{j=1}^{d} -ls_j * (I(x_1j != x_2j)))``
    Parameters
    -----------
    * `length_scale` [float, array-like, shape=[n_features,], 1.0 (default)]
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    * `length_scale_bounds` [array-like, [1e-5, 1e5] (default)]
        The lower and upper bound on length_scale
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1
        if anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        * `X` [array-like, shape=(n_samples_X, n_features)]
            Left argument of the returned kernel k(X, Y)
        * `Y` [array-like, shape=(n_samples_Y, n_features) or None(default)]
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        * `eval_gradient` [bool, False(default)]
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        * `K` [array-like, shape=(n_samples_X, n_samples_Y)]
            Kernel k(X, Y)
        * `K_gradient` [array-like, shape=(n_samples_X, n_samples_X, n_dims)]
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1

        if np.iterable(length_scale):
            if len(length_scale) > 1:
                length_scale = np.asarray(length_scale, dtype=np.float)
            else:
                length_scale = float(length_scale[0])
        else:
            length_scale = float(length_scale)

        X = np.atleast_2d(X)
        if anisotropic and X.shape[1] != len(length_scale):
            raise ValueError(
                "Expected X to have %d features, got %d" %
                (X.shape, len(length_scale)))

        n_samples, n_dim = X.shape

        Y_is_None = Y is None
        if Y_is_None:
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        indicator = np.expand_dims(X, axis=1) != Y
        kernel_prod = np.exp(-np.sum(length_scale * indicator, axis=2))

        # dK / d theta = (dK / dl) * (dl / d theta)
        # theta = log(l) => dl / d (theta) = e^theta = l
        # dK / d theta = l * dK / dl

        # dK / dL computation
        if anisotropic:
            grad = (-np.expand_dims(kernel_prod, axis=-1) *
                    np.array(indicator, dtype=np.float32))
        else:
            grad = -np.expand_dims(kernel_prod * np.sum(indicator, axis=2),
                                   axis=-1)

        grad *= length_scale
        if eval_gradient:
            return kernel_prod, grad
        return kernel_prod

class WeightedHammingKernel(StationaryKernelMixin, NormalizedKernelMixin,
                    Kernel):
    """
    The HammingKernel is used to handle categorical inputs.
    ``K(x_1, x_2) = exp(\sum_{j=1}^{d} -ls_j * (I(x_1j != x_2j)))``
    Parameters
    -----------
    * `length_scale` [float, array-like, shape=[n_features,], 1.0 (default)]
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    * `length_scale_bounds` [array-like, [1e-5, 1e5] (default)]
        The lower and upper bound on length_scale
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1
        if anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        * `X` [array-like, shape=(n_samples_X, n_features)]
            Left argument of the returned kernel k(X, Y)
        * `Y` [array-like, shape=(n_samples_Y, n_features) or None(default)]
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        * `eval_gradient` [bool, False(default)]
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        * `K` [array-like, shape=(n_samples_X, n_samples_Y)]
            Kernel k(X, Y)
        * `K_gradient` [array-like, shape=(n_samples_X, n_samples_X, n_dims)]
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1

        if np.iterable(length_scale):
            if len(length_scale) > 1:
                length_scale = np.asarray(length_scale, dtype=np.float)
            else:
                length_scale = float(length_scale[0])
        else:
            length_scale = float(length_scale)

        X = np.atleast_2d(X)
        if anisotropic and X.shape[1] != len(length_scale):
            raise ValueError(
                "Expected X to have %d features, got %d" %
                (X.shape, len(length_scale)))

        n_samples, n_dim = X.shape

        Y_is_None = Y is None
        if Y_is_None:
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        #import pdb; pdb.set_trace()
        param_types = np.array(['categorical' if isinstance(x, str) else 'numeric' for x in X[0]])
        numerical_idxs = np.where(np.array(param_types)=='numeric')[0]
        categorical_idxs = np.where(np.array(param_types)=='categorical')[0]

        indicator = np.expand_dims(X[:, categorical_idxs], axis=1) != Y[:, categorical_idxs]
        categorical_part = np.array(-np.sum(length_scale * indicator, axis=2), dtype='float')
        squared_diff = (np.expand_dims(X[:, numerical_idxs], axis=1) - Y[:, numerical_idxs])**2
        continuous_part = np.array(-np.sum(length_scale * squared_diff, axis=-1), dtype='float')
        kernel_prod = np.exp(categorical_part + continuous_part)

        # dK / d theta = (dK / dl) * (dl / d theta)
        # theta = log(l) => dl / d (theta) = e^theta = l
        # dK / d theta = l * dK / dl

        # dK / dL computation
        if anisotropic:
            grad = (-np.expand_dims(kernel_prod, axis=-1) *
                    np.expand_dims((categorical_part + continuous_part), axis=-1))
        else:
            grad = np.expand_dims(kernel_prod * (categorical_part + continuous_part),
                                   axis=-1)
        grad *= length_scale
        if eval_gradient:
            return kernel_prod, grad
        return kernel_prod