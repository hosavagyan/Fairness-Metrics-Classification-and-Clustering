from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test

def model_gaus(X_train, X_test, y_train, y_test):
    # Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    y_pred_gaus = gaussian.predict(X_test)
    probs_gaus = gaussian.predict_proba(X_test)
    score_gaussian = gaussian.score(X_test, y_test)
    print('The accuracy of Gaussian Naive Bayes is', score_gaussian)
    return gaussian,y_pred_gaus, probs_gaus,'gaus'

def model_svc(X_train, X_test, y_train, y_test):
    # Support Vector Classifier (SVM/SVC)
    svc = SVC(gamma=0.22, probability=True)
    svc.fit(X_train, y_train)
    y_pred_svm = svc.predict(X_test)
    probs_svc = svc.predict_proba(X_test)
    score_svc = svc.score(X_test, y_test)
    print('The accuracy of SVC is', score_svc)
    return svc, y_pred_svm, probs_svc, 'smv'

def model_logistic(X_train, X_test, y_train, y_test):
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    probs_lr = logreg.predict_proba(X_test)
    score_logreg = logreg.score(X_test, y_test)
    print('The accuracy of the Logistic Regression is', score_logreg)
    return logreg, y_pred_lr, probs_lr, 'lr'

def model_rf(X_train, X_test, y_train, y_test):
    # Random Forest Classifier
    randomforest = RandomForestClassifier()
    randomforest.fit(X_train, y_train)
    y_pred_rf = randomforest.predict(X_test)
    probs_rf = randomforest.predict_proba(X_test)
    score_randomforest = randomforest.score(X_test, y_test)
    print('The accuracy of the Random Forest Model is', score_randomforest)
    return randomforest, y_pred_rf, probs_rf, 'rf'

def model_knn(X_train, X_test, y_train, y_test):
    # K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    probs_knn = knn.predict_proba(X_test)
    score_knn = knn.score(X_test, y_test)
    print('The accuracy of the KNN Model is', score_knn)
    return knn, y_pred_knn, probs_knn, 'knn'

def model_pipline(X_train, X_test, y_train, y_test):

    model_list = [
        model_gaus,
        model_svc, # SVM runs long
        model_logistic,
        model_rf,
        model_knn
    ]

    for model in model_list:
        yield model(X_train, X_test, y_train, y_test)

