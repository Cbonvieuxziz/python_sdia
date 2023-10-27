import numpy as np


def markov(rho,A,nmax,rng, N):
    '''
    Build a Markov chain

    Params:
        rho: law of the initial state (probability vector)
        A: transition matrix (of size NxN) 
        nmax: number of time steps
        rng: random number generator
        X: trajectory of the chain
    
        returns:
            The path obtained
    '''
    
    def check_parameters(rho, A):
        '''
        Assert the input parameters are valid
        '''
        assert rho.shape == (N,), f"rho has to be a vector of N={N} elements"
        assert np.isclose(sum(rho), 1) and np.all(rho >= 0), "rho have to be in the unit simplex"
        assert A.shape == (N, N), f"A have to be of size NxN ({N}x{N})"
        assert np.all(A>=0) and np.allclose(np.sum(A, axis=1), np.ones((N,))), "A has to be stochastic"
    
    # Check if the parameters match the constraints
    check_parameters(rho, A)

    # Initialization
    X_0 = rng(N, rho)
    path = [X_0]
    
    for _ in range(nmax):                
        rho = A[X_0, :]        
        X_0 = rng(N, rho)  # X_(n+1)
        path.append(X_0)
    
    return path


def rng(N, rho):
    '''
    Random generator
    '''
    return np.random.choice(a=np.arange(N), p=rho)