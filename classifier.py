from sklearn.base import BaseEstimator

def make_fake_prob(predicted):
    fake_prob = np.zeros((predicted.shape[0], 4))
    for i, j in enumerate(float_labels(predicted)):
        fake_prob[i, j] = 1
    return fake_prob

class Classifier(BaseEstimator):
    def __init__(self):
        self.svc = SVC(C=1e7, kernel='rbf', probability=True)
        self.svc_fft = SVC(C=1e7, kernel='rbf', probability=True)
        self.bst = XGBClassifier(max_depth=3, learning_rate=0.3, n_estimators=200,
                            objective='multi:softprob')
        self.bst_fft = XGBClassifier(max_depth=3, learning_rate=0.3, n_estimators=200,
                                objective='multi:softprob')
        self.final_svc = SVC(C=1000)
        # self.final_svc.support_ = np.array([...])
        # self.final_svc.support_vectors_ = np.array([...])
        # ...
        
    def fit(self, X, y):
        """
        We fit the 4 corresponding models
        """
        X, X_fft = X
        self.svc.fit(X, y)
        self.svc_fft.fit(X_fft, y)
        self.bst.fit(X, y)
        self.bst_fft.fit(X_fft, y)

    def predict(self, X):
        pass

    def predict_proba(self, X):
        """
        We compute the probabilities and use our final SVC classifier
        to give the output. Since the output is a hard value and not
        a probability, we have to fake probabilities use make_fake_prob
        """
        X, X_fft = X
        probas = np.hstack([
            self.svc.predict_proba(X),
            self.svc_fft.predict_proba(X_fft),
            self.bst.predict_proba(X),
            self.bst_fft.predict_proba(X_fft),
        ])

        return make_fake_prob(self.final_svc.predict(probas))