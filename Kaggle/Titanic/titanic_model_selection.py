# %%
from sklearn.ensemble import VotingClassifier

# VotingClassifier approach

best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft') 
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft') 
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train_array,y_train_array,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train_array,y_train_array,cv=5).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train_array,y_train_array,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train_array,y_train_array,cv=5).mean())

print('voting_clf_all :',cross_val_score(voting_clf_all,X_train_array,y_train_array,cv=5))
print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train_array,y_train_array,cv=5).mean())

print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train_array,y_train_array,cv=5))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train_array,y_train_array,cv=5).mean())

# Making predictions
voting_clf_hard.fit(X_train_array, y_train_array)
voting_clf_soft.fit(X_train_array, y_train_array)
voting_clf_all.fit(X_train_array, y_train_array)
voting_clf_xgb.fit(X_train_array, y_train_array)
best_rf.fit(X_train_array, y_train_array)

y_hat_vc_hard = voting_clf_hard.predict(X_test_array).astype(int)
y_hat_rf = best_rf.predict(X_test_array).astype(int)
y_hat_vc_soft =  voting_clf_soft.predict(X_test_array).astype(int)
y_hat_vc_all = voting_clf_all.predict(X_test_array).astype(int)
y_hat_vc_xgb = voting_clf_xgb.predict(X_test_array).astype(int)

# Converting output to dataframe
final_data = {'PassengerId': test.PassengerId, 'Survived': y_hat_rf}
submission = pd.DataFrame(data=final_data)

final_data_2 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_hard}
submission_2 = pd.DataFrame(data=final_data_2)

final_data_3 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_soft}
submission_3 = pd.DataFrame(data=final_data_3)

final_data_4 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_all}
submission_4 = pd.DataFrame(data=final_data_4)

final_data_5 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_xgb}
submission_5 = pd.DataFrame(data=final_data_5)
