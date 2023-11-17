# %%
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

# Turning each dataframe into numpy array for cross validation
X_train_array=X_train_scaled.to_numpy()
y_train_array=y_train.to_numpy()
X_test_array=X_test_scaled.to_numpy()

# Each model
gnb=GaussianNB()
cv=cross_val_score(gnb, X_train_array, y_train_array, cv=5)
print('GaussianNB')
print(cv)
print(cv.mean()) # 0.72217

lr=LogisticRegression(max_iter=2000)
cv=cross_val_score(lr, X_train_array, y_train_array, cv=5)
print('LogisticRegression')
print(cv)
print(cv.mean()) # 0.82231

dt=tree.DecisionTreeClassifier(random_state=1)
cv=cross_val_score(dt, X_train_array, y_train_array, cv=5)
print('DecisionTreeClassifier')
print(cv)
print(cv.mean()) # 0.77619

knn=KNeighborsClassifier()
cv=cross_val_score(knn, X_train_array, y_train_array, cv=5)
print('KNeighborsClassifier')
print(cv)
print(cv.mean()) # 0.81895

rf=RandomForestClassifier(random_state=1)
cv=cross_val_score(rf, X_train_array, y_train_array, cv=5)
print('RandomForestClassifier')
print(cv)
print(cv.mean()) # 0.80318

svc=SVC(probability=True)
cv=cross_val_score(svc, X_train_array, y_train_array, cv=5)
print('SVC')
print(cv)
print(cv.mean()) # 0.83243

xgb=XGBClassifier()
cv=cross_val_score(xgb, X_train_array, y_train_array, cv=5)
print('XGBClassifier')
print(cv)
print(cv.mean()) # 0.81101

voting_clf=VotingClassifier(estimators=[('lr',lr), ('knn', knn), ('rf',rf), ('gnb', gnb), ('svc', svc), ('xgb', xgb)], voting='soft')
cv=cross_val_score(voting_clf, X_train_array, y_train_array, cv=5)
print('VotingClassifier')
print(cv)
print(cv.mean()) # 0.82568

# One example of prediction - may change voting_clf into any other models
voting_clf.fit(X_train_array, y_train_array)
y_hat_base_vc=voting_clf.predict(X_test_array).astype(int)
basic_submission={'PassengerId':test.PassengerId, 'Survived':y_hat_base_vc}
base_submission=pd.DataFrame(data=basic_submission)
