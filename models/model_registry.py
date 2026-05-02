from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


def get_models():
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC_linear": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
        "SVC_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "SVC_poly": SVC(kernel="poly", degree=3, probability=True, random_state=42),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
