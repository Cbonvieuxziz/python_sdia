import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cpdef distances(cnp.ndarray[cnp.double_t, ndim=1] x, cnp.ndarray[cnp.double_t, ndim=2] mat):
    cdef int N = mat.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=2] extended_x = np.tile(x, (N, 1))
    cdef cnp.ndarray[cnp.double_t, ndim=1] distances = np.linalg.norm(extended_x - mat, axis=1)
    return distances

cpdef K_Nearest(int K, str mode, cnp.ndarray[cnp.double_t, ndim=2] x_test, cnp.ndarray[cnp.double_t, ndim=2] x_train, cnp.ndarray[cnp.int32_t, ndim=1] class_train):
    cdef int N = x_test.shape[0]
    cdef int M = x_test.shape[1]
    cdef cnp.ndarray[cnp.int32_t, ndim=2] id = np.empty((N, K), dtype=np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=2] labels
    cdef cnp.ndarray[cnp.int32_t, ndim=1] inferred = np.empty(N, dtype=np.int32)
    cdef int k, n

    if K > x_train.shape[0]:
        K = x_train.shape[0]

    if K <= 0:
        raise ValueError("K must be a positive integer")

    if mode != "classification" and mode != "regression":
        raise ValueError("Mode must be 'classification' or 'regression'")

    # Pre-allocate memory for distances
    cdef cnp.ndarray[cnp.double_t, ndim=2] x_test_distances = np.empty((N, x_train.shape[0]), dtype=np.double_t)

    for n in range(N):
        for k in range(x_train.shape[0]):
            dist = 0.0
            for m in range(M):
                dist += (x_test[n, m] - x_train[k, m]) ** 2
            x_test_distances[n, k] = sqrt(dist)

    for n in range(N):
        sorted_indices = np.argsort(x_test_distances[n, :])
        id[n, :] = sorted_indices[:K]
        labels = class_train[id[n, :]].astype(np.int32)

        if mode == "classification":
            inferred[n] = np.argmax(np.bincount(labels))
        elif mode == "regression":
            inferred[n] = np.mean(labels)

    return inferred