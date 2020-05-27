import numpy as np 
import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy
from sklearn import tree
import graphviz
import pydotplus
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

def datapreprocessingstringtonumber(data):
	
	# Business Travel
	# Business_Travel = data.BusinessTravel.unique()
	aBusiness_Travel=["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
	bBusiness_Travel=[0,1,2]
	data = data.replace(aBusiness_Travel,bBusiness_Travel)
	# print(data.BusinessTravel)

	#Department
	# department_names = data.Department.unique()
	adepartment = ["Sales", "Human Resources", "Research & Development"]
	bdepartment = [0,1,2]
	data = data.replace(adepartment, bdepartment)
	# print(data.Department)

	#EducationField
	# EducationField_names = data.EducationField.unique()
	aEducationField = ['Medical','Human Resources','Life Sciences','Marketing','Technical Degree', 'Other',]
	bEducationField = [0,1,2,3,4,5]
	data = data.replace(aEducationField,bEducationField)
	# print(data.EducationField)	

	# Gender
	# Gender_names = data.Gender.unique()
	aGender_names = ["Male", "Female"]
	bGender_names = [0,1]
	data = data.replace(aGender_names,bGender_names)
	# print(data.Gender)	

	# JobRole
	# JobRole_names = data.JobRole.unique()
	aJobRole = ['Laboratory Technician', 'Sales Representative', 'Manager', 'Human Resources', 'Healthcare Representative', 'Sales Executive', 'Manufacturing Director', 'Research Scientist', 'Research Director']
	bJobRole = [0,1,2,3,4,5,6,7,8]
	data = data.replace(aJobRole,bJobRole)
	# print(data.JobRole)	
	
	# MaritalStatus
	# MaritalStatus_names = data.MaritalStatus.unique()
	aMaritalStatus = ['Single', 'Married', 'Divorced']
	bMaritalStatus = [0, 1, 2]
	data = data.replace(aMaritalStatus, bMaritalStatus)
	# print(data.MaritalStatus)	

	# OverTime
	# OverTime_names = data.OverTime.unique()
	aOverTime = ['No', 'Yes']
	bOverTime = [0, 1]
	data = data.replace(aOverTime, bOverTime)
	# print(data.OverTime)	
	return data



def normalise(dataToN):
	mean = dataToN.mean(axis = 0)
	std = dataToN.std(axis = 0)
	for column in dataToN:
		dataToN[column]=(dataToN[column]-mean[column])/std[column]
	return dataToN

scalar = StandardScaler()


##################*******main_function********###################
data = pd.read_csv("train.csv")
data = datapreprocessingstringtonumber(data)
data = data.drop(['EmployeeCount', 'EmployeeNumber', 'ID'], axis = 1)
data = data.drop(['MonthlyIncome'], axis = 1);

##For PCA
# scalar.fit(data)
# data = scalar.transform(data)

y = data['Attrition']
x = data.drop(['Attrition'], axis = 1)

##Normalisation
x = normalise(x)
x_train, y_train = x, y;


# #Logistic Regression
# clf = LogisticRegression(max_iter = 1000)
# clf.fit(x_train, y_train)
# y_predict = clf.predict(x_validation)
# comp = np.array(y_predict) == np.array(y_validation)
# accuracy = np.count_nonzero(comp)/np.size(comp)
# print(accuracy)
 


# # Linear SVC
# clf = LinearSVC(random_state=0, max_iter = 10000)
# clf.fit(x_train, y_train)
# # print(clf.coef_)
# # print(clf.intercept_)
# y_predict = clf.predict(x_validation)
# comp = np.array(y_predict) == np.array(y_validation)
# accuracy = np.count_nonzero(comp)/np.size(comp)
# print("validation", accuracy)

# y_predict1 = clf.predict(x_train)
# comp = np.array(y_predict1) == np.array(y_train)
# accuracy1 = np.count_nonzero(comp)/np.size(comp)
# print("train", accuracy1)


# #decision tree
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x_train, y_train)
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph=pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("d_tree.png")
# # graph.write_pdf("d_tree.pdf")
# y_predict = clf.predict(x_validation)
# comp = np.array(y_predict) == np.array(y_validation)
# accuracy = np.count_nonzero(comp)/np.size(comp)
# print(accuracy)

# ## Random Forest
# clf = RandomForestClassifier(max_depth=4, random_state=0)
# clf.fit(x_train, y_train)
# print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train)))


# # # #gradient boosting
learning_rate = 0.11
clf = GradientBoostingClassifier(n_estimators=70, learning_rate=learning_rate, random_state=0)
clf.fit(x_train, y_train)
# , 


#############****making output excel files****############ 
data_test = pd.read_csv("test.csv")
data_test = datapreprocessingstringtonumber(data_test)
ID = data_test.ID

# scalar.fit(data_test)
# scalar.transform(data_test)

data_test = data_test.drop(['EmployeeCount', 'EmployeeNumber', 'ID'], axis = 1) ####### remove ID from feature
data_test = data_test.drop(['MonthlyIncome'] , axis = 1);
data_test = normalise(data_test)
predictions = clf.predict(data_test) 

dict = {'ID': ID, 'Attrition': predictions}
df = pd.DataFrame(dict, columns= ['ID', 'Attrition']) 
df.to_csv('output.csv', index=False)





###############analysing Correlation
# print(data.corr())


# plt.figure(figsize = (35, 30))
# sns.heatmap(data.corr())
# plt.show()

# print(scipy.stats.spearmanr(data.PerformanceRating, data.PercentSalaryHike))


##########################
