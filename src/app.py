from utils import db_connect
engine = db_connect()

# your code here
#Loading the dataset
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')
df_raw

# info del dataset:
df_raw.info()

# descripción de las variables:
df_raw.describe()

df_raw.hist(figsize=(12,12))
plt.show()

df_raw = df_raw[(df_raw["BMI"] > 0 ) & (df_raw["BloodPressure"] > 0) & (df_raw["Glucose"] > 0)]
df_filter = df_raw.copy()
df_filter

# separacion de variable:
X= df_filter.iloc[:, :8]
y= df_filter.iloc[:, 8]

# Separo la data en train y en test: 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

# modelo de train simple 
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# mejor modelo 
clf1 = DecisionTreeClassifier(criterion='entropy',
                              splitter='best',
                             min_samples_split=20,
                             min_samples_leaf=5,
                             random_state=0, max_depth=3)

clf1.fit(X_train, y_train)
print('Accuracy:',clf1.score(X_test, y_test))

#show predicted dataset
clf1_pred=clf1.predict(X_test)

cm = confusion_matrix(y_test, clf1_pred, labels=clf1.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf1.classes_)
disp.plot()

plt.show()

plt.figure(figsize=(10,8))
plot_tree(clf1)
plt.show()

