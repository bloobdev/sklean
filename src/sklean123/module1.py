"""
Hello



"""

class Number(object):

    def __init__(self, n):
        self.value = n

    def val(self):
        return self.value

    def add(self, n2):
        self.value += n2.val()

    def __add__(self, n2):
        return self.__class__(self.value + n2.val())

    def __str__(self):
        return str(self.val())

    @classmethod
    def addall(cls, number_obj_iter):
        cls(sum(n.val() for n in number_obj_iter))


def help():
    """
    This is a custom help message for kill yourself


Decision tree - I

Importing Required Libraries

    # Load libraries
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    from sklearn.model_selection import train_test_split # Import train_test_split function
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation/evaluation

Loading Data

    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset
    pima = pd.read_csv("diabetes.csv", header=None, names=col_names)

    pima.head(10)

    pima.drop([0,8],axis=0, inplace = True)

    pima.head(10)

Feature Selection

    #split dataset in features and target variable
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
    X = pima[feature_cols] # Features
    y = pima.label # Target variable

Splitting Data

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 70% training and 20% test

Building Decision Tree Model

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    y_pred

    y_test

Evaluating Model

    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    cm = metrics.confusion_matrix(y_test,y_pred)
    print("Confusion matrix \n", cm)

    ###############################################################################################################################

Decision tree - II

Data Preprocessing Step

    # importing libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # importing datasets
    data_set = pd.DataFrame(pd.read_csv("D:/ISDC/SJCC Data Mining with Python/SJCC Class 3/suv_data.csv"))
    print(data_set)

    data_set.info()

    # Extracting Independent and dependent variables
    x = data_set.iloc[:,[2,3]].values # selecting Age and Estimated Salary as our predictors
    y = data_set.iloc[:,4].values # Selecting purchased as target variable

    # splitting the dataset into training and test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

    # feature scaling
    from sklearn.preprocessing import StandardScaler
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    print(max(x_train[:,0]))
    print(min(x_train[:,0]))

    print(max(x_train[:,1]))
    print(min(x_train[:,1]))

Fitting the Decision Tree algorithm to the training set

    # Fitting the decision tree classifier to the training set
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(x_train, y_train)

Predicting the test set result

    # Predicting the test set result
    y_pred = classifier.predict(x_test)
    y_train_pred = classifier.predict(x_train)

    y_pred

    y_train_pred

    # Creating the confusion matrix for training data
    from sklearn.metrics import confusion_matrix
    cm_train = confusion_matrix(y_train,y_train_pred)
    print(cm_train)

    cm = confusion_matrix(y_test,y_pred)
    print(cm)

    # Visualising the training set result
    from matplotlib.colors import ListedColormap
    x_set, y_set = x_train, y_train
    x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step=0.01),
                        np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
    plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                alpha=0.75, cmap=ListedColormap(('purple','green')))
    plt.xlim(x1.min(),x1.max())
    plt.ylim(x2.min(),x2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                   c = ListedColormap(('white','black'))(i),label=j)
    plt.title('Decision Tree Algorithm (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    # Visualising the test set result
    from matplotlib.colors import ListedColormap
    x_set, y_set = x_test, y_test
    x1, x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1, step=0.01),
                        np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step=0.01))
    plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                alpha=0.75, cmap=ListedColormap(('purple','green')))
    plt.xlim(x1.min(),x1.max())
    plt.ylim(x2.min(),x2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set==j,0], x_set[y_set==j,1],
                   c = ListedColormap(('white','black'))(i),label=j)
    plt.title('Decision Tree Algorithm (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    ###############################################################################################################################

Decision tree - III

    # importing libraries
    import numpy as np
    import matplotlib.pyplot as plt

    dataset = np.array([['Asset Flip',100,1000],
                       ['Text Based', 500,3000],
                       ['Visual Novel',1500,5000],
                       ['2D Pixel Art',3500,8000],
                       ['2D Vector Art',5000,6500],
                       ['Strategy',6000,7000],
                       ['First Person Shooter',8000,15000],
                       ['Simulator',9500,20000],
                       ['Racing',12000,21000],
                       ['RPG',14000,25000],
                       ['Sandbox',15500,27000],
                       ['Open-World',16500,30000],
                       ['MMOFPS',25000,52000],
                       ['MMORPG',30000,80000]])

    # select all rows by : and column 1 by 1:2 representing features
    X = dataset[:,1:2].astype(int)
    print(X)

    # select all rows by : and column 2 by 2 to y representing labels
    y = dataset[:,2].astype(int)
    print(y)

    # import the regressor
    from sklearn.tree import DecisionTreeRegressor

    # Create a regressor object
    regressor = DecisionTreeRegressor(random_state = 0)

    # fit the regressor with X and y data
    regressor.fit(X,y)

    # predicting a new value
    # test the output by changing values, like 3750
    y_pred = regressor.predict([[4850]])

    # print the predicted price
    print("Predicted price: % d\n"% y_pred)

    r_sq = regressor.score(X,y)
    print(f"Coefficient of determination: {r_sq}")

    # arrange for creating a range of values from min value of X to
    # max value of X with a difference of 0.01 between two consecutive values
    X_grid = np.arange(min(X), max(X), 0.01)

    X_grid

    # reshaping the data into a len(X_grid)*1 array. i.e. to make a column
    # out of the X_grid values
    X_grid = X_grid.reshape((len(X_grid),1))

    X_grid

    # Scatter plot for original data
    plt.scatter(X,y, color='red')

    # plot predicted data
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')

    # specify title
    plt.title("Profit to production cost (Decision Tree Regression)")

    plt.xlabel("Production cost")
    plt.ylabel("Profit")
    plt.show()

    from sklearn.tree import export_graphviz

    # export the decision tree to a tree.dot file
    # for visualising the plot easily anywhere
    export_graphviz(regressor, out_file = 'tree1.dot',
                   feature_names = ['Production Cost'])


SVM - I

    # Loading the dataset
    import pandas as pd

    # Reading Iris dataset as pandas data frame
    df = pd.DataFrame(pd.read_csv("iris.csv"))

    df.head()

    df.shape

    print(df.info())

    # Seperating dependent and independent variables

    x = df.iloc[:,0:-1]
    y = df.iloc[:,-1:]

    # Splitting into train and test datasets

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0, stratify = y)

    # Feature scaling

    from sklearn.preprocessing import StandardScaler
    st_x = StandardScaler()

    x_train = st_x.fit_transform(x_train)
    x_test = st_x.fit_transform(x_test)

    # Fitting the SVM Classifier

    from sklearn.svm import SVC

    classifier = SVC(kernel= 'linear', random_state=0, max_iter = (500))
    classifier.fit(x_train,y_train)

    # Predicting
    y_pred = classifier.predict(x_test)

    # Checking model accuracy through confusion matrix
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

    # Confusion matrix for test data
    print(confusion_matrix(y_test,y_pred))

    # Accuracy score of model for test data
    print(accuracy_score(y_test,y_pred))
    print(classifier.score(x_test,y_test))

    cr = classification_report(y_test,y_pred)
    print(cr)

SVM - II

    # Step 1 : Loading the Dataset
    import pandas as pd

    #Reading mtcars data as pandas Data Frame
    df=pd.DataFrame(pd.read_csv("mtcars.csv"))

    #Treating Null Values
    print(df.info())
    df=df.dropna(axis=0)

    # Step 2 : Seperating dependent and independent variables
    X=df.iloc[:,2:]   
    y=df.iloc[:,1]

    # Step 3 : split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        random_state=42)

    # Step 4 : Create and Train Model

    # create Support Vector Regression (SVR) model
    from sklearn.svm import SVR

    # Create a instance of SVR using RBF 
    svr = SVR(kernel='rbf', C=1, gamma='scale', epsilon=0.1)

    # train the model
    svr.fit(X_train, y_train)

    # predict on the test set
    y_pred = svr.predict(X_test) 
    y_pred_tr = svr.predict(X_train)

    # Step 5 : Evaluate the model using mean squared error
    from sklearn.metrics import mean_squared_error,r2_score
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: ", mse)
    print(r2_score(y_test, y_pred)) 

    #In this code, we first load the dataset and split it into training and testing 
    #sets using train_test_split function from sklearn. Then, we create an instance 
    #of the SVR class and specify the kernel type, regularization parameter C, 
    #kernel coefficient gamma, and the parameter for error tolerance epsilon. 
    #After creating the model, we fit it on the training data using the fit method. 
    #Finally, we predict the target variable for the test set using the predict 
    #method and evaluate the performance of the model using the mean squared error 
    #metric from the metrics module.

SVM - III

    # Linear SVM

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import svm

    # linear data
    X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
    y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

    # show unclassified data
    plt.scatter(X, y)
    plt.show()

    # shaping data for training the model
    training_X = np.vstack((X, y)).T
    training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

    # define the model
    clf = svm.SVC(kernel='linear', C=1.0)

    clf.fit(training_X, training_y)

    # get the weight values for the linear equation from the trained SVM model
    w = clf.coef_[0]

    # get the y-offset for the linear equation
    a = -w[0] / w[1]

    # make the x-axis space for the data points
    XX = np.linspace(0, 13)

    # get the y-values to plot the decision boundary
    yy = a * XX - clf.intercept_[0] / w[1]

    # plot the decision boundary
    plt.plot(XX, yy, 'k-')

    # show the plot visually
    plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
    plt.show()

    # Non-linear SVM

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets
    from sklearn import svm

    # non-linear data
    circle_X, circle_y = datasets.make_circles(n_samples=300, noise=0.05)

    # show raw non-linear data
    plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, marker='.')
    plt.show()

    # make non-linear algorithm for model
    nonlinear_clf = svm.SVC(kernel='rbf', C=1.0)

    # training non-linear model
    nonlinear_clf.fit(circle_X, circle_y)

    # Plot the decision boundary for a non-linear SVM problem
    def plot_decision_boundary(model, ax=None):
        if ax is None:
            ax = plt.gca()
            
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)

        # shape data
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        
        # get the decision boundary based on the model
        P = model.decision_function(xy).reshape(X.shape)
        
        # plot decision boundary
        ax.contour(X, Y, P,
                   levels=[0], alpha=0.5,
                   linestyles=['-'])

    # plot data and decision boundary
    plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, s=50)
    plot_decision_boundary(nonlinear_clf)
    plt.scatter(nonlinear_clf.support_vectors_[:, 0], nonlinear_clf.support_vectors_[:, 1], s=50, lw=1, facecolors='none')
    plt.show()

SVM - IV

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2] # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    y = iris.target

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0 # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=1,gamma=0.1).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
     np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()

    svc = svm.SVC(kernel='rbf', C=100,gamma=0.1).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
     np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()


#####################################################################################################################################
K-means clustering - Implementation 2

    # importing libraries
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import MinMaxScaler
    import seaborn as sns
    import matplotlib.pyplot as plt
    %matplotlib inline



    train = pd.read_csv("titanic_train.csv")
    test = pd.read_csv("titanic_test.csv")

    print("***** Train_Set *****")
    print(train.describe())
    print("\n")
    print("***** Test_Set *****")
    print(test.describe())

    # check for NA values

    # for the train set
    train.isna().head()

    print("***** In the train set *****")
    print(train.isna().sum())
    print("\n")
    print("***** In the test set *****")
    print(test.isna().sum())

    # fill missing values with mean column values in the train set
    train.fillna(train.mean(), inplace=True)
    print(train.isna().sum())

    # fill missing values with mean column values in the test set
    test.fillna(test.mean(), inplace=True)
    print(test.isna().sum())

    train = train.drop(['Name','Ticket','Cabin','Embarked'], axis = 1)
    test = test.drop(['Name','Ticket','Cabin','Embarked'], axis = 1)

    labelEncloder = LabelEncoder()
    labelEncloder.fit(train['Sex'])
    train["Sex"] = labelEncloder.transform(train["Sex"])
    test["Sex"] = labelEncloder.transform(test["Sex"])

    train.info(); test.info()

    # model
    X = np.array(train.drop(["Survived"],1).astype(float))
    y = np.array(train["Survived"])

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    # calculate the silhouette_score
    from sklearn.metrics import silhouette_score

    print(silhouette_score(X, kmeans.labels_))

    #! pip install yellowbrick

    from yellowbrick.cluster import KElbowVisualizer

    model = KMeans(random_state=0)
    visualizer = KElbowVisualizer(model, k=(2,6),
                                metric='silhouette',
                                timings = False)

    # fit the data and visualize
    visualizer.fit(X)

    visualizer.poof()

####################################################
K-Means Clustering using sklearn

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set() #for plot styling
    import sklearn.datasets as data

    # from sklearn.datasets.samples_generator import make blobs
    from sklearn.datasets import make_blobs

    X,y_true = data.make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=0)
    plt.scatter(X[:,0],X[:,1],c=y_true,cmap="rainbow")

    ## Elbow method

    from sklearn.cluster import KMeans

    wcss = []
    for i in range(1,11):
        km = KMeans(n_clusters=i)
        km.fit_predict(X)
        wcss.append(km.inertia_)

    wcss

    plt.plot(range(1,11),wcss)
    plt.xlabel("No. of clusters")
    plt.ylabel("wcss")

    from sklearn.cluster import KMeans
    kmean=KMeans(n_clusters=4)
    kmean.fit(X)
    y_kmean = kmean.predict(X)

    plt.scatter(X[:,0],X[:,1],c=y_kmean,cmap='viridis')
    centers = kmean.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1],c='red',s=200,alpha=0.5)

    #Limitations of K-Means Algorithm
    from sklearn.datasets import make_moons
    X, y = make_moons(200, noise=.05, random_state=0)
    plt.scatter(X[:, 0], X[:, 1], c = y,cmap='rainbow');

    labels = KMeans(2, random_state=0).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis');

    #Comparision with some other Non-Convex Clustering Algorithm
    from sklearn.cluster import SpectralClustering
    import warnings
    warnings.simplefilter("ignore")

    model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',assign_labels='kmeans')
    labels = model.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis');
#######################################################
DBSCAN - Density Based Spatial Clustering for Applicariton with Noise

    # importing necessary libraries
    import numpy as np
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    np.random.seed(42)

    # function to create points in a circle
    def CirclePoints(r, n=200, noise=20):
        return [(math.cos(2*math.pi/n*i)*r+np.random.uniform(-noise,noise),
                math.sin(2*math.pi/n*i)*r+np.random.uniform(-noise,noise)) for i in range(n)]

    # Creating 1st circle
    df = pd.DataFrame(CirclePoints(100,200,20))
    plt.scatter(df[0], df[1], s=10)

    # Creating 2nd Circle and adding to previous sets of points
    df = pd.concat([df,pd.DataFrame(CirclePoints(300,400,40))])
    plt.scatter(df[0], df[1], s=10)

    # Creating 3rd circle and adding to previous sets of points
    df = pd.concat([df,pd.DataFrame(CirclePoints(500,700,60))])
    plt.scatter(df[0],df[1],s=10)

    # Creating random noise points and adding to previous sets points
    df = pd.concat([df, pd.DataFrame([(np.random.randint(-650,650),
                                    np.random.randint(-650,650)) for i in range(200)])])
    plt.scatter(df[0], df[1], s=10)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df[[0,1]])

    df["KMeans"] = kmeans.labels_

    plt.figure(figsize=(8,8))
    plt.scatter(df[0], df[1], c=df["KMeans"])

    # Using DBSCAN without arguments
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN()
    dbscan.fit(df[[0,1]])

    df["DBSCAN"] = dbscan.labels_
    plt.figure(figsize=(8,8))
    plt.scatter(df[0], df[1], c= df["DBSCAN"])

    # Using DBSCAN with arguments
    dbscan = DBSCAN(eps=50, min_samples=10)
    dbscan.fit(df[[0,1]])

    df["DBSCAN"] = dbscan.labels_
    plt.figure(figsize=(8,8))
    plt.scatter(df[0], df[1], c=df["DBSCAN"])
######################################################
Agglomerative clustering - Implementation 2


    # importing libraries
    import numpy as np

    X = np.array([[5,3],
                [10,15],
                [15,12],
                [24,10],
                [30,30],
                [85,70],
                [71,80],
                [60,78],
                [70,55],
                [80,91],])

    import matplotlib.pyplot as plt

    labels = range(1,11)
    plt.figure(figsize=(10,7))
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(X[:,0],X[:,1], label='True Position')

    for label, x,y in zip(labels, X[:,0], X[:,1]):
        plt.annotate(label,
                    xy=(x,y), xytext=(-3,3),
                    textcoords = 'offset points', ha = 'right', va = 'bottom')
        plt.show()

    # dendogram
    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib import pyplot as plt

    linked = linkage(X,'single')

    labelList = range(1,11)

    plt.figure(figsize=(10,7))
    dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()
##############################################################################
Agglomerative Clustering - Implementation-Copy1

    # importing libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # reading the dataset
    df = pd.read_csv("shopping data.csv")

    print(df.shape)
    print(df.head())

    df.info()

    # filtering data required for clustering
    data = df.iloc[:,3:5].values

    # hierarchy clustering
    import scipy.cluster.hierarchy as sch

    plt.figure(figsize=(15,8))
    dend = sch.dendrogram(sch.linkage(data, method='ward'))
    plt.axhline(y=120)

    # agglomerative clustering
    from sklearn.cluster import AgglomerativeClustering
    clusters = AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='complete')

    clusters.fit_predict(data)

    plt.figure(figsize=(15,8))
    plt.scatter(data[:,0], data[:,1],
            c=clusters.labels_,
            cmap="rainbow")
#############################################################################
Agglomerative Clustering - Implementation

    # importing libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # reading the dataset
    df = pd.read_csv("shopping data.csv")

    print(df.shape)
    print(df.head())

    # filtering data required for clustering
    data = df.iloc[:,3:5].values

    # hierarchy clustering
    import scipy.cluster.hierarchy as sch

    plt.figure(figsize=(15,8))
    dend = sch.dendrogram(sch.linkage(data, method='ward'))
    plt.axhline(y=120)

    # agglomerative clustering
    from sklearn.cluster import AgglomerativeClustering
    clusters = AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='complete')

    clusters.fit_predict(data)

    plt.figure(figsize=(15,8))
    plt.scatter(data[:,0], data[:,1],
            c=clusters.labels_,
            cmap="rainbow")

    # advantage over k-means
    from sklearn.datasets import make_moons, make_circles

    dummy_data = make_moons(n_samples=500,
                        shuffle=True,
                        noise=0.09,
                        random_state=40)
    dummy_data1 = make_circles(n_samples=1000,
                            shuffle=True,
                            noise=0.03,
                            random_state=40,
                            factor=0.6)

    plt.scatter(dummy_data1[0][:,0], dummy_data1[0][:,1],
            c=dummy_data1[1])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dummy_data1[0])
    plt.scatter(dummy_data1[0][:,0],dummy_data1[0][:,1], c= kmeans.labels_)

    from sklearn.cluster import AgglomerativeClustering
    clusters = AgglomerativeClustering(n_clusters=2,
                                    affinity="euclidean",
                                    linkage="single")

    clusters.fit_predict(dummy_data1[0])
    plt.scatter(dummy_data1[0][:,0], dummy_data1[0][:,1], c=clusters.labels_)
###########################################################

    
Question Bank 
    What is a decision tree? 
    a.	A tree with leaves representing the decision made by the algorithm 
    b.	A graph with branches representing the decisions made by the algorithm 
    c.	A chart that displays data in a hierarchical structure 
    d.	None of the above 
    Answer: b 
    
    What is a decision tree regressor? 
    a.	A decision tree that predicts continuous values 
    b.	A decision tree that predicts categorical values 
    c.	A decision tree that predicts both continuous and categorical values 
    d.	None of the above 
    Answer: a 
    
    Which type of machine learning algorithm is Classification and Regression Tree (CART)? a. Unsupervised Learning 
    b.	Supervised Learning 
    c.	Reinforcement Learning 
    d.	Semi-supervised Learning 
    Answer: b 
    
    What is pruning in the context of Decision Trees? 
    a.	A technique for removing features from the dataset 
    b.	technique for reducing the complexity of the model 
    c.	A technique for increasing the accuracy of the model 
    d.	A technique for handling missing values 
    Answer: b 
        
    In a Classification and Regression Tree (CART), what is the primary objective of splitting the data into subsets? 
    a.	To make the model more complex 
    b.	To reduce the accuracy of the model 
    c.	To maximize the variance in each subset 
    d.	To increase the homogeneity within each subset 
    Answer: d 
        
    What type of variables can be used in a Classification and Regression Tree (CART)? 
    a.	Only continuous variables 
    b.	Only categorical variables 
    c.	Both continuous and categorical variables 
    d.	Discrete variables 
    Answer: c 
    
    What is a decision tree classifier? 
    a.	A decision tree that predicts continuous values 
    b.	A decision tree that predicts categorical values 
    c.	A decision tree that predicts both continuous and categorical values 
    d.	None of the above 
    Answer: b 
    
    How does a decision tree work? 
    a.	It splits the dataset into subsets based on the value of a feature 
    b.	It predicts the target variable based on the features of the dataset 
    c.	It calculates the entropy of the dataset to determine the best split 
    d.	All of the above 
    Answer: a 
    
    What is the goal of a decision tree? 
    a.	To maximize the accuracy of the predictions 
    b.	To minimize the complexity of the model 
    c.	To maximize the information gain at each split 
    d.	All of the above 
    Answer: d 
    
    What is entropy in the context of decision trees? 
    a.	A measure of the disorder or randomness of a dataset 
    b.	A measure of the accuracy of the predictions made by the tree 
    c.	A measure of the complexity of the tree 
    d.	None of the above 
    Answer: a 
    
    What is information gain in the context of decision trees? 
    a.	The reduction in entropy after a dataset is split on a feature 
    b.	The increase in entropy after a dataset is split on a feature 
    c.	The accuracy of the predictions made by the tree 
    d.	None of the above 
    Answer: a 
    
    How does a decision tree handle missing values? 
    a.	It assigns the most common value of the feature to the missing values 
    b.	It assigns the mean value of the feature to the missing values 
    c.	It uses a surrogate feature to predict the missing values 
    d.	All of the above 
    Answer: d 
    
    How does a decision tree handle categorical features? 
    a.	It converts them into numerical values 
    b.	It creates a binary feature for each possible value 
    c.	It uses a one-hot encoding to represent the categorical values 
    d.	All of the above 
    Answer: d 
    
    What is pruning in the context of decision trees? 
    a.	Removing unnecessary branches from the tree to reduce complexity and overfitting 
    b.	Adding extra branches to the tree to increase accuracy 
    c.	Assigning more weight to certain features in the tree 
    d.	None of the above 
    Answer: a 
    
    What is overfitting in the context of decision trees? 
    a.	When the tree is too simple and does not capture the complexity of the data 
    b.	When the tree is too complex and fits the noise in the data instead of the signal 
    c.	When the tree is unable to predict the target variable accurately 
    d.	None of the above 
    Answer: b 
    
    What is underfitting in the context of decision trees? 
    a.	When the tree is too simple and does not capture the complexity of the data 
    b.	When the tree is too complex and fits the noise in the data instead of the signal 
    c.	When the tree is unable to predict the target variable accurately 
    d.	None of the above 
    Answer: a 
    
    What is a node in a decision tree? 
    a)	A decision or a condition that splits the data into subsets. 
    b)	The final prediction or classification made by the tree. 
    c)	The root of the tree. 
    Answer: a
    What is a leaf node in a decision tree? 
    a)	A decision or a condition that splits the data into subsets. 
    b)	The final prediction or classification made by the tree. 
    c)	The root of the tree. 
    Answer: a
    What is a feature in a decision tree? 
    a)	The target variable. 
    b)	The variables used to make decisions or splits in the tree. 
    c)	The data used to train the model. 
    Answer: b

    
    What is clustering? 
    a)	A technique used to classify objects into categories. 
    b)	A technique used to reduce the dimensionality of data. 
    c)	A technique used to make predictions about new data. 
    Answer: a) A technique used to classify objects into categories. 
    
    What is the objective of clustering? 
    a)	To group similar objects together. 
    b)	To identify patterns in the data. 
    c)	To predict the target variable. 
    Answer: a) To group similar objects together. 
    
    What is the difference between clustering and classification? 
    a)	Clustering is unsupervised while classification is supervised. 
    b)	Clustering is used for continuous variables while classification is used for categorical variables. 
    c)	Clustering is used to find patterns while classification is used to make predictions. 
    Answer: a) Clustering is unsupervised while classification is supervised. 
    
    What is the difference between hierarchical and partitioning clustering? 
    a)	Hierarchical clustering is used for categorical variables while partitioning clustering is used for continuous variables. 
    b)	Hierarchical clustering creates a tree-like structure while partitioning clustering creates a flat structure. 
    c)	Hierarchical clustering is supervised while partitioning clustering is unsupervised. 
    Answer: b) Hierarchical clustering creates a tree-like structure while partitioning clustering creates a flat structure. 
    
    What is K-means clustering? 
    a)	A type of partitioning clustering. 
    b)	A type of hierarchical clustering. 
    c)	A type of density-based clustering. 
    Answer: a) A type of partitioning clustering. 
    
    What is the objective of K-means clustering? 
    a)	To minimize the distance between the centroids and the data points. 
    b)	To maximize the distance between the centroids and the data points. 
    c)	To group the data points into a predetermined number of clusters. 
    Answer: a) To minimize the distance between the centroids and the data points. 
    
    What is the elbow method in K-means clustering? 
    a)	A technique used to select the number of clusters. 
    b)	A technique used to measure the distance between the centroids and the data points. 
    c)	A technique used to visualize the clustering results. 
    Answer: a) A technique used to select the number of clusters. 
    
    What is the objective of clustering? 
    a)	To create a predictive model that can classify or predict the outcome of a given data point 
    b)	To identify patterns and group similar data points together based on similarity 
    c)	To identify the most important features in a dataset 
    d)	To reduce the dimensionality of a dataset 
    Answer: b) To identify patterns and group similar data points together based on similarity 
    
    
    What is the difference between K-Means and Hierarchical Clustering? 
    a)	K-Means is faster and more accurate than Hierarchical Clustering 
    b)	Hierarchical Clustering is faster and more accurate than K-Means 
    c)	K-Means is a non-parametric algorithm, while Hierarchical Clustering is a parametric algorithm 
    d)	K-Means requires the user to specify the number of clusters, while Hierarchical Clustering does not 
    Answer: d) K-Means requires the user to specify the number of clusters, while Hierarchical Clustering does not 
    
    What is a dendrogram in the context of Agglomerative Clustering? 
    a)	A visualization of the distance between each pair of data points 
    b)	A visualization of the number of clusters required to obtain a given level of similarity 
    c)	A visualization of the hierarchy of clusters obtained by the algorithm 
    d)	A visualization of the silhouette score for different clustering solutions 
    Answer: c) A visualization of the hierarchy of clusters obtained by the algorithm 
    
    What is the Silhouette score in clustering? 
    a)	A measure of how well the data points are separated into clusters. 
    b)	A measure of how well the centroids are placed. 
    c)	A measure of how well the clustering algorithm is performing. 
    Answer: a) A measure of how well the data points are separated into clusters. 
    
    What is the difference between centroid-based and density-based clustering? 
    a)	Centroid-based clustering creates a tree-like structure while density-based clustering creates a flat structure. 
    b)	Centroid-based clustering assumes that clusters have a spherical shape while density-based clustering can handle clusters of any shape. 
    c)	Centroid-based clustering is unsupervised while density-based clustering is supervised. 
    Answer: b) Centroid-based clustering assumes that clusters have a spherical shape while densitybased clustering can handle clusters of any shape. 
    
    What is DBSCAN? 
    a)	A type of centroid-based clustering. 
    b)	A type of density-based clustering. 
    c)	A type of hierarchical clustering. 
    Answer: b) A type of density-based clustering. 
    
    What is the objective of DBSCAN? 
    a)	To minimize the distance between the centroids and the data points. 
    b)	To maximize the distance between the centroids and the data points. 
    c)	To group the data points into clusters of high density. 
    Answer: c) To group the data points into clusters of high density. 
    
    
    What is the disadvantage of Agglomerative Clustering? 
    a)	It is computationally expensive 
    b)	It can only handle categorical variables 
    c)	It is prone to producing unbalanced clusters 
    d)	It requires a pre-defined number of clusters 
    Answer: a) It is computationally expensive 
    
    What is the purpose of backpropagation in ANN? 
    A.	To adjust the weights of the neural network during training 
    B.	To propagate signals forward through the network 
    C.	To adjust the learning rate of the network 
    D.	None of the above 
    Answer: A. To adjust the weights of the neural network during training 
    
    Which type of learning is used to train ANNs using examples? 
    A.	Supervised learning 
    B.	Unsupervised learning 
    C.	Reinforcement learning 
    D.	All of the above 
    Answer: A. Supervised learning 
    
    What is the primary purpose of ANNs? 
    A.	To perform classification tasks 
    B.	To perform regression tasks 
    C.	To perform both classification and regression tasks 
    D.	None of the above 
    Answer: C. To perform both classification and regression tasks 
    
    Which of the following is an advantage of using ANNs over traditional algorithms? 
    A.	ANNs can learn from data 
    B.	ANNs are faster than traditional algorithms 
    C.	ANNs are easier to program than traditional algorithms 
    D.	None of the above 
    Answer: A. ANNs can learn from data 
    
    Which of the following is a disadvantage of using ANNs? 
    A.	ANNs are computationally expensive 
    B.	ANNs are not as accurate as traditional algorithms 
    C.	ANNs require large amounts of data to train D. None of the above 
    Answer: A. ANNs are computationally expensive 
    
    Which layer of an ANN is responsible for making predictions? 
    A.	Input layer 
    B.	Hidden layer 
    C.	Output layer 
    D.	None of the above 
    Answer: C. Output layer 
    
    Which of the following is an example of a feedforward neural network? 
    A.	Recurrent neural network 
    B.	Convolutional neural network 
    C.	Perceptron 
    D.	None of the above 
    Answer: C. Perceptron 
    
    Which of the following is an example of a recurrent neural network? 
    A.	Perceptron 
    B.	Convolutional neural network 
    C.	Long short-term memory (LSTM) 
    D.	None of the above 
    Answer: C. Long short-term memory (LSTM) 
    
    Which type of learning is used to train ANNs without labeled data? 
    A.	Supervised learning 
    B.	Unsupervised learning 
    C.	Reinforcement learning 
    D.	None of the above 
    Answer: B. Unsupervised learning 
    
    What is the role of the activation function in a neuron? 
    A.	To normalize the output of the neuron to a specific range 
    B.	To introduce non-linearity into the output of the neuron 
    C.	To adjust the weights of the connections between neurons 
    D.	To compute the error between the predicted and actual outputs of the neuron Answer: B. To introduce non-linearity into the output of the neuron 
    Answer: B
    
    What is overfitting in the context of ANNs? 
    A.	When the model is too simple and fails to capture the complexity of the data 
    B.	When the model is too complex and fits the noise in the data instead of the underlying patterns 
    C.	When the model has too few neurons and cannot learn the underlying patterns in the data D. When the model has too many neurons and cannot generalize to new data 
    Answer: B. When the model is too complex and fits the noise in the data instead of the underlying patterns 
    
    What is the basic unit of an artificial neuron in an ANN? 
    A.	Input 
    B.	Output 
    C.	Activation Function 
    D.	Weight 
    Answer: C. Activation Function 
    
    What is the purpose of the hidden layers in an ANN? 
    A.	To provide an additional input to the output layer 
    B.	To provide a feedback loop to the input layer 
    C.	To perform feature engineering on the input data 
    D.	To increase the complexity of the model and improve its performance 
    Answer: D. To increase the complexity of the model and improve its performance 
    
    Which of the following is an example of unsupervised learning in ANNs? 
    A.	K-means clustering 
    B.	Decision tree 
    C.	Random forest 
    D.	None of the above 
    Answer: A. K-means clustering 
    
    Which of the following is an example of reinforcement learning in ANNs? 
    A.	Q-learning 
    B.	Gradient descent 
    C.	Backpropagation 
    D.	None of the above 
    Answer: A. Q-learning 
    
    
    What does SVM stand for? 
    A.	Support Vector Machine 
    B.	Simple Vector Model 
    C.	Singular Value Method 
    D.	Sequential Vector Mapping 
    Answer: A. Support Vector Machine 
    
    What is the main objective of SVM? 
    A.	To find the maximum margin hyperplane that separates the data 
    B.	To find the minimum margin hyperplane that separates the data 
    C.	To classify the data into different groups 
    D.	None of the above 
    Answer: A. To find the maximum margin hyperplane that separates the data 
    
    What is the kernel function in SVM? 
    A.	It is a function used to transform the input data into a higher-dimensional space 
    B.	It is a function used to transform the output data into a lower-dimensional space 
    C.	It is a function used to normalize the input data 
    D.	None of the above 
    Answer: A. It is a function used to transform the input data into a higher-dimensional space 
    
    Which of the following is a popular kernel function used in SVM? 
    A.	Linear kernel 
    B.	Polynomial kernel 
    C.	RBF kernel 
    D.	All of the above 
    Answer: D. All of the above 
    
    Which type of SVM is used for binary classification problems? 
    A.	Linear SVM 
    B.	Nonlinear SVM 
    C.	One-class SVM 
    D.	None of the above 
    Answer: A. Linear SVM 
    
    Which type of SVM is used for multi-class classification problems? 
    A.	Linear SVM 
    B.	Nonlinear SVM 
    C.	One-class SVM 
    D.	None of the above 
    Answer: B. Nonlinear SVM 
    
    Which of the following is an advantage of using SVM? 
    A.	It is effective in high-dimensional spaces 
    B.	It is less computationally expensive than other algorithms 
    C.	It can handle non-linear decision boundaries 
    D.	None of the above 
    Answer: A. It is effective in high-dimensional spaces 
    
    Which of the following is a disadvantage of using SVM? 
    A.	It is sensitive to the choice of kernel function 
    B.	It can be computationally expensive for large datasets 
    C.	It can overfit the data if the margin is too small 
    D.	All of the above 
    Answer: D. All of the above 
    
    Which of the following is used to determine the optimal hyperplane in SVM? 
    A.	Margin 
    B.	Support vectors 
    C.	Both margin and support vectors 
    D.	None of the above 
    Answer: C. Both margin and support vectors 
    
    What is the goal of SVM? 
    A.	To maximize the margin between the decision boundary and the data points 
    B.	To minimize the number of misclassifications 
    C.	To maximize the number of correctly classified data points 
    D.	To minimize the number of support vectors 
    Answer: A. To maximize the margin between the decision boundary and the data points 
    
    What is a kernel function in the context of SVM? 
    A.	A function used to transform the input data to a higher dimensional space 
    B.	A function used to regularize the weights in the model 
    C.	A function used to initialize the weights in the model 
    D.	A function used to reduce the dimensionality of the input data 
    Answer: A. A function used to transform the input data to a higher dimensional space 
    
    What is the role of support vectors in SVM? 
    A.	They define the decision boundary of the SVM 
    B.	They are the data points closest to the decision boundary 
    C.	They are the data points with the highest weight in the model 
    D.	They are the data points with the lowest weight in the model 
    Answer: B. They are the data points closest to the decision boundary 
    
    What is the difference between a linear SVM and a nonlinear SVM? 
    A.	A linear SVM can only separate the data with a straight line, while a nonlinear SVM can separate the data with a curved boundary 
    B.	A linear SVM can only handle two-class problems, while a nonlinear SVM can handle multiclass problems 
    C.	A linear SVM can only handle numerical data, while a nonlinear SVM can handle both numerical and categorical data 
    D.	A linear SVM is faster and less complex than a nonlinear SVM 
    Answer: A
    
    What is the margin in SVM? 
    A.	It is the distance between the hyperplane and the nearest data point 
    B.	It is the distance between the hyperplane and the farthest data point 
    C.	It is the distance between the hyperplane and the mean of the data 
    D.	None of the above 
    Answer: A. It is the distance between the hyperplane and the nearest data point 
    
    Which of the following is used to measure the quality of the hyperplane in SVM? 
    A.	Accuracy 
    B.	Precision 
    C.	Recall 
    D.	F1-score 
    Answer: D. F1-score 
    
    Which of the following is an example of a linear SVM? 
    A.	Soft-margin SVM 
    B.	Hard-margin SVM 
    C.	Kernel SVM 
    D.	None of the above 
    Answer: B. Hard-margin SVM 
    
    Which of the following is an example of a nonlinear SVM? 
    A.	Soft-margin SVM 
    B.	Hard-margin SVM 
    C.	Kernel SVM 
    D.	None of the above 
    Answer: C. Kernel SVM 
    
    What is Market Basket Analysis (MBA)? 
    A.	It is a data analysis technique that identifies the association between products frequently purchased together 
    B.	It is a technique for predicting future stock market trends 
    C.	It is a technique for analyzing sales trends of a company 
    D.	None of the above 
    Answer: A. It is a data analysis technique that identifies the association between products frequently purchased together 
    
    What is a market basket? 
    A.	It is the basket used by customers to carry their purchases in a store 
    B.	It is a collection of items purchased together in a single transaction 
    C.	It is the list of items available for purchase in a store 
    D.	None of the above 
    Answer: B. It is a collection of items purchased together in a single transaction 
    
    What is the purpose of market basket analysis? 
    A.	To understand customer behavior and buying patterns 
    B.	To increase store revenue 
    C.	To improve customer experience 
    D.	All of the above 
    Answer: D. All of the above 
    
    What is the objective of Market Basket Analysis? 
    A.	To identify the most profitable products in a company's product line 
    B.	To identify the items that are frequently purchased together by customers 
    C.	To identify the most popular products in a market 
    D.	To identify the products that have the highest profit margins 
    Answer: B. To identify the items that are frequently purchased together by customers 
    
    What is the support metric in Market Basket Analysis? 
    A.	A measure of how frequently the item set appears in the transaction data 
    B.	A measure of the strength of association between the items in the item set 
    C.	A measure of the lift between the items in the item set 
    D.	A measure of the confidence between the items in the item set 
    Answer: A. A measure of how frequently the item set appears in the transaction data 
    
    What is the lift metric in Market Basket Analysis? 
    A.	The ratio of the support of the itemset to the support of the individual items 
    B.	The ratio of the support of the itemset to the support of the complement of the itemset 
    C.	The ratio of the support of the itemset to the total number of transactions in the dataset D. The probability that the itemset will be purchased together 
    Answer: A. The ratio of the support of the itemset to the support of the individual items 
    
    What is market basket analysis? 
    A.	A statistical technique to identify the most popular items in a store 
    B.	A technique to analyze customer buying behavior and identify relationships between products 
    C.	A method to optimize store layout and product placement 
    D.	A tool to track inventory levels in a store 
    Answer: B. A technique to analyze customer buying behavior and identify relationships between products 
    
    Which of the following is the measure of association used in market basket analysis? A. Confidence 
    B.	Lift 
    C.	Support 
    D.	All of the above 
    Answer: D. All of the above 
    
    What is support in market basket analysis? 
    A.	It is the probability of an item being purchased in a transaction 
    B.	It is the probability of two items being purchased together in a transaction 
    C.	It is the ratio of transactions containing an item to the total number of transactions 
    D.	None of the above 
    Answer: C. It is the ratio of transactions containing an item to the total number of transactions 
    
    What is confidence in market basket analysis? 
    A.	It is the probability of an item being purchased given that another item is purchased 
    B.	It is the probability of two items being purchased together in a transaction 
    C.	It is the ratio of transactions containing both items to the total number of transactions containing the first item 
    D.	None of the above 
    Answer: A. It is the probability of an item being purchased given that another item is purchased 
    
    What is lift in market basket analysis? 
    A.	It is the ratio of the probability of both items being purchased together to the product of their individual probabilities 
    B.	It is the ratio of the probability of one item being purchased to the probability of the other item being purchased 
    C.	It is the ratio of the number of transactions containing both items to the total number of transactions 
    D.	None of the above 
    Answer: A. It is the ratio of the probability of both items being purchased together to the product of their individual probabilities 
    
    What does a lift value greater than 1 indicate in market basket analysis? 
    A.	There is a positive association between the items 
    B.	There is a negative association between the items 
    C.	There is no association between the items 
    D.	None of the above 
    Answer: A. There is a positive association between the items 
    
    Which of the following is an algorithm used for market basket analysis? 
    A.	Apriori algorithm 
    B.	Decision tree algorithm 
    C.	K-means algorithm 
    D.	None of the above 
    Answer: A. Apriori algorithm 
    
    What is the minimum support threshold in the Apriori algorithm? 
    A.	It is the minimum number of items required in a transaction for it to be included in the analysis 
    B.	It is the minimum probability required for an item to be considered frequent 
    C.	It is the minimum number of transactions containing an item required for it to be considered frequent 
    D.	None of the above 
    Answer: C. It is the minimum number of transactions containing an item required for it to be considered 
        
        
    
    Usage:
    my_function()

    Returns:
    A string with a custom message.
    """
    return "This is a custom response."