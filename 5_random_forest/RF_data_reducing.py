#Function transform is deprecated; Support to use estimators as feature selectors will be removed in version 0.19. Use SelectFromModel instead.


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X_train = [
        [0,1,2],
        [3,5,4],
        [6,8,7],
        [9,11,10],
        [12,14,13],
        [15,16,17],
        [18,19,20],
        [21,22,23],
]

Y_train = [0,0,0,0,1,1,1,1]

rf=RandomForestClassifier(n_estimators=400, n_jobs=-1)
sf=SelectFromModel(rf)

Xr_train=sf.fit_transform(X_train,Y_train)

print(Xr_train)