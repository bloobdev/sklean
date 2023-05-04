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
    
    Usage:
    my_function()

    Returns:
    A string with a custom message.
    """
    return "This is a custom response."