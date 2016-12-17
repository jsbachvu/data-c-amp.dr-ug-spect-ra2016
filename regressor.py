class Regressor(BaseEstimator):
    """
    There are 3 SVC that classify
    svc_split split the data between concentration < 3000 and concentration >= 3000
    svc_small is trained only on molecules that have a concentration < 3000
    svc_big is trained only on molecules that have a concentration >= 3000

    At predicting time, we look at if
    """
    def __init__(self, split=3000):
        self.split = split
        self.svc_split = SVC(C=1e7)
        self.svc_small = SVC(C=1e7)
        self.svc_big = SVC(C=1e7)

    def fit(self, X, y):
        # the indices corresponding to where training label is small
        is_small = np.array(y < self.split).astype(np.int)

        small = np.array(y < self.split)
        # train the splitting SVC on the whole data
        self.svc_split.fit(X, is_small)
        # train the small SVC on the data corresponding to small concentrations
        self.svc_small.fit(X[small], y[small])
        # train the big SVC on the data corresponding to big concentrations
        self.svc_big.fit(X[~small], y[~small])

    def predict(self, X):
        res = np.zeros(X.shape[0])
        
        # predicted indices that correspond to small concentration
        is_small = self.svc_split.predict(X).astype(np.bool)
        
        # we predicted the data that is supposed to correpond to small 
        # concentration with the small SVC
        small_pred = self.svc_small.predict(X[is_small])
        # same for data that is supposed to be big with big SVC
        big_pred = self.svc_big.predict(X[~is_small])
        
        # we update our response
        res[is_small] = small_pred
        res[~is_small] = big_pred

        return res