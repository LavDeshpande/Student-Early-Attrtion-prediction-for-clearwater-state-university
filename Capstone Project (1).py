#!/usr/bin/env python
# coding: utf-8

# In[1193]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.preprocessing import StandardScaler


# In[1194]:


df=pd.read_excel("/Users/lav/Downloads/Student Applications & Performance.xlsx")


# # Data Preprocessing

# **Null Values-(columns-nulls-column_length-datatype)

# In[1195]:


for i in df.columns:
    print(i," ",df[i].isnull().sum(),"  ", len(df["STDNT_AGE"]),"  ", df[i].dtype)


# # Getting Rid of the Null Values

# In[1196]:


columns=["SECOND_TERM_EARNED_HRS","SECOND_TERM_ATTEMPT_HRS","MOTHER_HI_EDU_CD","FATHER_HI_EDU_CD","HIGH_SCHL_GPA"
        ,"DISTANCE_FROM_HOME","STDNT_TEST_ENTRANCE_COMB"]


# In[1197]:


for i in columns:
    df[i]=df[i].replace(np.nan,df[i].mean())


# # Dealing with Categorial Variables

# In[1198]:


columns=["CORE_COURSE_GRADE_1_F","CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F"]


# In[1199]:


for i in columns:
    df[i]=df[i].replace({"A":5,"B":4,"C":3,"D":2,"F":1,"INCOMPL":0,"NOT REP":0})


# In[1200]:


columns=["CORE_COURSE_GRADE_4_F"]
for i in columns:
    df[i]=df[i].replace({"A":5,"B":4,"C":3,"D":2,"F":1,"INCOMPL":0})


# In[1201]:


columns=["CORE_COURSE_GRADE_5_F","CORE_COURSE_GRADE_6_F"]
for i in columns:
    df[i]=df[i].replace({"A":5,"B":4,"C":3,"D":2})


# In[1202]:


columns=["CORE_COURSE_GRADE_2_S","CORE_COURSE_GRADE_3_S"]
for i in columns:
    df[i]=df[i].replace({"A":5,"B":4,"C":3,"D":2,"F":1,"INCOMPL":0})
columns=["CORE_COURSE_GRADE_1_S"]
for i in columns:
    df[i]=df[i].replace({"A":5,"B":4,"C":3,"D":2,"F":1,"INCOMPL":0,"NOT REP":0,"Unknown":0})
columns=["CORE_COURSE_GRADE_4_S","CORE_COURSE_GRADE_5_S"]
for i in columns:
    df[i]=df[i].replace({"A":5,"B":4,"C":3,"D":2,"F":1})


# **Hence we may drop CORE_COURSE_GRADE_6_S,CORE_COURSE_GRADE_6_F,CORE_COURSE_GRADE_5_F,CORE_COURSE_GRADE_6_S

# # Binning Categorical Variables and Performimg PCA

# In[1049]:


columns=["CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_1_F","CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F"
        ,"CORE_COURSE_GRADE_4_F","CORE_COURSE_GRADE_5_F","CORE_COURSE_GRADE_6_F"
        ,"CORE_COURSE_GRADE_2_S","CORE_COURSE_GRADE_3_S","CORE_COURSE_GRADE_4_S"
        ,"CORE_COURSE_GRADE_5_S"]


# In[1203]:


a=(df["RETURNED_2ND_YR"].value_counts()[0])/(df["RETURNED_2ND_YR"].value_counts()[0]+df["RETURNED_2ND_YR"].value_counts()[1])


# In[1204]:


df["STDNT_BACKGROUND"].value_counts()


# In[1205]:


df_dropped=df[df["RETURNED_2ND_YR"]==0]
df_continued=df[df["RETURNED_2ND_YR"]==1]


# In[1206]:


categorial_variables=[]
for i in df.columns:
    if len(df[i].value_counts())<=10:
        categorial_variables.append(i)


# In[1207]:


categorial_variables


# In[1212]:


df_continued["CORE_COURSE_GRADE_6_F"].value_counts()


# In[1209]:


del categorial_variables[4]
del categorial_variables[11]
del categorial_variables[19]                     


# In[1215]:


print("Ratio of no of returned students vs no of non-returned students in dataset -",a)
for j in categorial_variables:
    print(j)
    try:     
        for i in range(len(df[j].value_counts())):
            drop_rate=df_dropped[j].value_counts()[i]/df[j].value_counts()[i]
            print(df[j].value_counts().index[i],"",drop_rate)
    except:
        try:
            for i in range (len(df[j].value_counts())-1):
                drop_rate=df_dropped[j].value_counts()[i+1]/df[j].value_counts()[i+1]
                print(df[j].value_counts().index[i],"",df[j].value_counts()[i+1],drop_rate)
        except:
            try:
                for i in range (len(df[j].value_counts()-2)):
                    drop_rate=df_dropped[j].value_counts()[i+2]/df[j].value_counts()[i+2]
                    print(df[j].value_counts().index[i],"",df[j].value_counts()[i+2],drop_rate)
            except:
                print("")


# In[1216]:


import seaborn as sns


# In[1217]:


sns.boxplot(df["RETURNED_2ND_YR"],df[df["DISTANCE_FROM_HOME"]<200]["DISTANCE_FROM_HOME"])


# In[1220]:


sns.boxplot(df["RETURNED_2ND_YR"],df[df["HIGH_SCHL_GPA"]>=2]["HIGH_SCHL_GPA"])


# In[1221]:


sns.boxplot(df["RETURNED_2ND_YR"],df[df["FIRST_TERM_EARNED_HRS"]>=2]["FIRST_TERM_EARNED_HRS"])


# In[1235]:


sns.boxplot(df["RETURNED_2ND_YR"],df[df["SECOND_TERM_ATTEMPT_HRS"]>=5]["SECOND_TERM_ATTEMPT_HRS"])


# In[1236]:


sns.boxplot(df["RETURNED_2ND_YR"],df[df["SECOND_TERM_EARNED_HRS"]>=6]["SECOND_TERM_EARNED_HRS"])


# In[1237]:


sns.boxplot(df["RETURNED_2ND_YR"],df["COST_OF_ATTEND"])


# In[1238]:


sns.boxplot(df["RETURNED_2ND_YR"],df["GROSS_FIN_NEED"])


# In[1245]:


sns.boxplot(df["RETURNED_2ND_YR"],df[df["EST_FAM_CONTRIBUTION"]<=1000000]["EST_FAM_CONTRIBUTION"])


# In[1246]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_1_F"])


# In[1247]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_2_F"])


# In[1248]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_3_F"])


# In[1249]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_4_F"])


# In[1233]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_1_S"])


# In[1253]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_1_S"])


# In[1252]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_2_S"])


# In[1254]:


sns.boxplot(df["RETURNED_2ND_YR"],df["CORE_COURSE_GRADE_3_S"])


# In[1259]:


sns.boxplot(df["RETURNED_2ND_YR"],df["STDNT_TEST_ENTRANCE1"])


# In[1261]:


sns.boxplot(df["RETURNED_2ND_YR"],df["STDNT_TEST_ENTRANCE2"])


# # Checking whether any of the student major has highest contribution in dropping out of the students

# In[1262]:


sto=df["STDNT_MAJOR"].value_counts()


# In[1263]:


stn=df_dropped["STDNT_MAJOR"].value_counts()


# In[1264]:


df_test=pd.DataFrame()


# In[1266]:


df_test["total"]=sto
df_test["dropped"]=stn


# In[1267]:


df_test["difference"]=df_test["total"]-df_test["dropped"]


# In[1268]:


df_test["drop_per"]=(df_test["dropped"]*100)/df_test["total"]


# In[1270]:


df_test=df_test.sort_values(by="drop_per",ascending=False)


# In[1279]:


df_test=df_test[df_test["drop_per"]>=25]


# In[1282]:


df_test[["drop_per"]].plot(kind="bar",figsize=(16,8),legend=True)


# # Transforming Student Major to Ordinal Category

# In[1284]:


_3=df_test[df_test["drop_per"]>=30]
_2=df_test[(df_test["drop_per"]>=20) & (df_test["drop_per"]<30)]
_1=df_test[df_test["drop_per"]<20]


# In[1285]:


_3=list(_3.index)
_2=list(_2.index)
_1=list(_1.index)


# # _3-Sub Majors with more than 30% student drop rate
# # _2-Sub Majors with 20-30 % student drop rate
# # _3-Sub Majors with below 20% student drop rate

# In[1292]:


_3


# In[1293]:


_2


# # PCA 

# In[1258]:


df=df.drop(columns=["CORE_COURSE_GRADE_6_S","CORE_COURSE_GRADE_6_F","CORE_COURSE_GRADE_5_S","CORE_COURSE_GRADE_5_F","CORE_COURSE_GRADE_4_F"])


# In[395]:


a=pd.get_dummies(df["STDNT_BACKGROUND"])


# In[396]:


df=pd.concat([df,a],axis=1)


# In[397]:


from sklearn.decomposition import PCA


# In[398]:


pca=PCA(n_components=1)


# In[399]:


p=pca.fit(df[["BGD 1","BGD 2","BGD 3","BGD 4","BGD 5","BGD 6","BGD 7","BGD 8"]])
p=pca.transform(df[["BGD 1","BGD 2","BGD 3","BGD 4","BGD 5","BGD 6","BGD 7","BGD 8"]])
pca.explained_variance_ratio_


# In[400]:


pca.components_


# In[401]:


df["Background"]=p


# In[402]:


df.drop(columns=["BGD 1","BGD 2","BGD 3","BGD 4","BGD 5","BGD 6","BGD 7","BGD 8"],inplace=True)


# In[403]:


df["IN_STATE_FLAG"]=df["IN_STATE_FLAG"].replace({"Y":1,"N":0})


# In[404]:


df["INTERNATIONAL_STS"]=df["INTERNATIONAL_STS"].replace({"Y":1,"N":0})


# **Female-1,Male-0

# In[405]:


df["STDNT_GENDER"]=df["STDNT_GENDER"].replace({"F":1,"M":0})


# In[406]:


df["DEGREE_GROUP_CD"].value_counts()


# In[407]:


df["DEGREE_GROUP_DESC"].value_counts()


# # Drop Column Degree Group

# In[408]:


df.drop(columns=["DEGREE_GROUP_DESC"],inplace=True)


# In[409]:


df.drop(columns=["DEGREE_GROUP_CD"],inplace=True)


# In[410]:


df["HOUSING_STS"].value_counts()


# In[411]:


df["HOUSING_STS"]=df["HOUSING_STS"].replace({"Off Campus":0,"On Campus":1})


# ** On Campus-1,Off Campus-0

# In[412]:


df_new=df.drop(columns=["MOTHER_HI_EDU_DESC","FATHER_HI_EDU_DESC","HIGH_SCHL_NAME","STDNT_BACKGROUND"])


# # Dropping Students Minor Column

# In[413]:


df_new.drop(columns=["STDNT_MINOR"],inplace=True)


# # Dropping Student Entrance 1 , 2 Columns

# In[414]:


df_new.drop(columns=["STDNT_TEST_ENTRANCE1","STDNT_TEST_ENTRANCE2"],inplace=True)


# # Binning the data in student major column

# In[425]:


df_new["STDNT_MAJOR"]=df_new["STDNT_MAJOR"].apply(lambda x : 3 if x in _3 else 2 if x in _2 else 1 )


# # Dropping the 1st term and 2nd term column

# In[426]:


df_new.drop(columns=["FIRST_TERM","SECOND_TERM"],inplace=True)


# In[427]:


df_new.drop(columns=["CORE_COURSE_NAME_6_F","CORE_COURSE_NAME_5_F","CORE_COURSE_NAME_6_S",
                    "CORE_COURSE_NAME_5_S","CORE_COURSE_NAME_4_S"],inplace=True)


# In[428]:


df_new.drop(columns=["CORE_COURSE_GRADE_4_S"],inplace=True)


# # Checking if is there any particalar course for which students are dropping out

# In[430]:


df_new["RETURNED_2ND_YR"].value_counts()


# In[431]:


df_new["CORE_COURSE_NAME_3_F"].value_counts()


# # There seems to be no such conclusion as the dropping percent for each course is nearly same as for the total dataset

# In[432]:


df_new.drop(columns=["CORE_COURSE_NAME_1_F","CORE_COURSE_NAME_2_F","CORE_COURSE_NAME_3_F",
            "CORE_COURSE_NAME_4_F","CORE_COURSE_NAME_1_S","CORE_COURSE_NAME_2_S",
            "CORE_COURSE_NAME_3_S"],inplace=True)


# In[433]:


df_new.drop(columns=["STUDENT IDENTIFIER"],inplace=True)


# In[434]:


X=df_new.drop(columns=["RETURNED_2ND_YR"])


# In[435]:


y=df_new["RETURNED_2ND_YR"]


# # EDA After Oversampling- Generating Systhetic Data

# In[616]:


from sklearn.model_selection import train_test_split


# In[436]:


columns=["CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F",
        "CORE_COURSE_GRADE_4_F","CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_2_S",
        "CORE_COURSE_GRADE_3_S"]


# In[437]:


for i in columns:
    X[i]=X[i].replace(np.nan,df[i].mean())


# In[834]:


sm=SMOTE(k_neighbors=5)


# In[844]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.08)


# In[845]:


X_train,X_test,Y_train,Y_test=train_test_split(x_train,y_train,test_size=0.15)


# In[846]:


X_train,Y_train=sm.fit_resample(X_train,Y_train)


# In[850]:


c=pd.concat([X_train,Y_train],axis=1)


# In[851]:


c_dropped=c[c["RETURNED_2ND_YR"]==0]


# In[852]:


c_continued=c[c["RETURNED_2ND_YR"]==1]


# # Distributions

# In[853]:


sns.distplot(c_dropped_s["COST_OF_ATTEND"])


# In[854]:


c


# In[855]:


sns.distplot(c_continued_s["COST_OF_ATTEND"])


# In[856]:


sns.distplot(c_dropped["COST_OF_ATTEND"])


# In[857]:


c["CORE_COURSE_GRADE_1_S"].value_counts()


# # The distribution is not a normal distribution

# # Anova Test

# In[866]:


columns=['STDNT_AGE',
 'STDNT_GENDER',
 'IN_STATE_FLAG',
 'INTERNATIONAL_STS',
 'STDNT_MAJOR',
 'STDNT_TEST_ENTRANCE_COMB',
 'CORE_COURSE_GRADE_1_F',
 'CORE_COURSE_GRADE_2_F',
 'CORE_COURSE_GRADE_3_F',
 'CORE_COURSE_GRADE_4_F',
 'CORE_COURSE_GRADE_1_S',
 'CORE_COURSE_GRADE_2_S',
 'CORE_COURSE_GRADE_3_S',
 'HOUSING_STS',
 'DISTANCE_FROM_HOME',
 'HIGH_SCHL_GPA',
 'FATHER_HI_EDU_CD',
 'MOTHER_HI_EDU_CD',
 'FIRST_TERM_ATTEMPT_HRS',
 'FIRST_TERM_EARNED_HRS',
 'SECOND_TERM_ATTEMPT_HRS',
 'SECOND_TERM_EARNED_HRS',
 'GROSS_FIN_NEED',
 'COST_OF_ATTEND',
 'EST_FAM_CONTRIBUTION',
 'UNMET_NEED',
 'Background']


# # Checking for Categorial and Continuous Variables

# In[867]:


Categorial_variables=[]
for i in columns:
    if len(c[i].value_counts())<10:
        Categorial_variables.append(i)
        print(c[i].value_counts())


# In[868]:


Categorial_variables


# In[869]:


Continuous_variables=[]
for i in columns:
    if len(c[i].value_counts())>=10:
        Continuous_variables.append(i)
        print(c[i].value_counts())


# In[870]:


Continuous_variables


# # Anova Test

# In[899]:


for i in Continuous_variables:
    a=c_dropped[i].sample(500)
    b=c_continued[i].sample(500)
    stat, p = f_oneway(a,b)
    if stat>=3:
        print(i,"","stat-",stat,"p_value-",p)


# In[932]:


from scipy.stats import kruskal


# In[943]:


for i in Categorial_variables:
    a=c_dropped[i].sample(100)
    b=c_continued[i].sample(100)
    stat, p = kruskal(a,b)
    if p<=0.08:
        print(i,"","stat-",stat,"p_value-",p)


# # Above parametes have different distributions for students who left Vs students who continued

# # Box Plot

# In[873]:


from seaborn import boxplot


# In[874]:


sns.boxplot(c["RETURNED_2ND_YR"],c[c["DISTANCE_FROM_HOME"]<200]["DISTANCE_FROM_HOME"])


# In[875]:


sns.boxplot(c["RETURNED_2ND_YR"],c[c["HIGH_SCHL_GPA"]>=2]["HIGH_SCHL_GPA"])


# In[876]:


sns.boxplot(c["RETURNED_2ND_YR"],c[c["FIRST_TERM_EARNED_HRS"]>=2]["FIRST_TERM_EARNED_HRS"])


# In[877]:


sns.boxplot(c["RETURNED_2ND_YR"],c[c["SECOND_TERM_ATTEMPT_HRS"]>=5]["SECOND_TERM_ATTEMPT_HRS"])


# In[878]:


sns.boxplot(c["RETURNED_2ND_YR"],c[c["SECOND_TERM_EARNED_HRS"]>=6]["SECOND_TERM_EARNED_HRS"])


# In[880]:


sns.boxplot(c["RETURNED_2ND_YR"],c["COST_OF_ATTEND"])


# In[881]:


sns.boxplot(c["RETURNED_2ND_YR"],c["GROSS_FIN_NEED"])


# In[882]:


sns.boxplot(c["RETURNED_2ND_YR"],c[c["EST_FAM_CONTRIBUTION"]<=1200000]["EST_FAM_CONTRIBUTION"])


# In[884]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_1_F"])


# In[886]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_2_F"])


# In[888]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_3_F"])


# In[890]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_4_F"])


# In[892]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_1_S"])


# In[894]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_2_S"])


# In[896]:


sns.boxplot(c["RETURNED_2ND_YR"],c["CORE_COURSE_GRADE_3_S"])


# # Standardizing

# In[909]:


columns=["CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F",
        "CORE_COURSE_GRADE_4_F","CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_2_S",
        "CORE_COURSE_GRADE_3_S"]


# In[910]:


for i in columns:
    X[i]=X[i].replace(np.nan,df[i].mean())


# In[911]:


sc=StandardScaler()


# In[912]:


X[X.columns]=sc.fit_transform(X[X.columns])


# In[913]:


from sklearn.model_selection import train_test_split


# In[914]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.08,random_state=0)


# In[915]:


X_train,X_test,Y_train,Y_test=train_test_split(x_train,y_train,test_size=0.15,random_state=0)


# # Over Sampling

# In[916]:


from imblearn.over_sampling import SMOTE


# In[917]:


sm=SMOTE(k_neighbors=6)


# In[918]:


X_train,Y_train=sm.fit_resample(X_train,Y_train)


# # K Neighbors

# In[65]:


from sklearn.neighbors import KNeighborsClassifier


# In[66]:


kn=KNeighborsClassifier(n_neighbors=5)


# In[67]:


kn.fit(X_train,Y_train)


# In[68]:


kn.score(X_test,Y_test)


# In[69]:


from sklearn.model_selection import GridSearchCV


# In[70]:


params={"n_neighbors":[3,5,6,7,9,11,13,15],"algorithm":["auto","brute"],"weights":["uniform","distance"]}


# In[71]:


kn=KNeighborsClassifier()


# In[72]:


cv=GridSearchCV(kn,params,scoring="roc_auc")


# In[73]:


cv.fit(X_train,Y_train)


# In[74]:


kn_model=cv.best_estimator_


# In[75]:


kn_model.fit(X_train,Y_train)


# In[76]:


kn_model.score(X_test,Y_test)


# In[80]:


from sklearn.metrics import confusion_matrix , classification_report , roc_curve , precision_recall_curve,roc_auc_score,precision_score,recall_score


# In[82]:


y_predict=kn_model.predict(X_test)


# In[83]:


confusion_matrix(Y_test,y_predict)


# # Dummy Classifier

# In[84]:


from sklearn.dummy import DummyClassifier


# In[85]:


dm=DummyClassifier(strategy="most_frequent")


# In[86]:


dm.fit(X_train,Y_train)


# In[87]:


dm.score(X_test,Y_test)


# In[88]:


y=dm.predict(X_test)


# In[89]:


confusion_matrix(Y_test,y)


# # Random Forest Classifier

# In[95]:


from sklearn.ensemble import RandomForestClassifier


# In[96]:


rfc=RandomForestClassifier(n_estimators=300,oob_score=True,max_depth=10,max_features=20,)


# In[97]:


rfc.fit(X_train,Y_train)


# In[98]:


rfc.score(X_test,Y_test)


# In[99]:


y_predict=rfc.predict(X_test)


# In[100]:


confusion_matrix(Y_test,y_predict)


# In[101]:


Y_test.value_counts()


# In[102]:


rfc.oob_score_


# In[103]:


precisions,recalls,thresholds=precision_recall_curve(Y_test,y_predict)


# In[104]:


plt.plot(precisions,recalls)


# In[105]:


fps,tps,thresholds=roc_curve(Y_test,y_predict)


# In[106]:


plt.plot(fps,tps)


# # Logistic Regression

# In[107]:


from sklearn.linear_model import LogisticRegression


# In[108]:


lr=LogisticRegression()


# In[109]:


params={"C":[1,2,3,4,6,8,9,12,15,20]}


# In[110]:


gs=GridSearchCV(lr,params)


# In[111]:


gs.fit(X_train,Y_train)


# In[112]:


lr_model=gs.best_estimator_


# In[113]:


lr_model.fit(X_train,Y_train)


# In[114]:


lr_model.score(X_test,Y_test)


# In[115]:


y_predict=lr_model.predict(X_test)


# In[116]:


confusion_matrix(Y_test,y_predict)


# In[117]:


roc_auc_score(Y_test,y_predict)


# In[118]:


precisions,recalls,thresholds=precision_recall_curve(Y_test,y_predict)


# In[119]:


plt.plot(precisions,recalls)


# In[120]:


fps,tps,thresholds=roc_curve(Y_test,y_predict)


# In[121]:


plt.plot(fps,tps)


# # Ensemble Techniques

# # Bagging Classifier

# In[122]:


from sklearn.ensemble import BaggingClassifier


# In[123]:


bag_clf=BaggingClassifier(lr_model,n_estimators=100,bootstrap=True)


# In[124]:


#bag_clf.fit(X_train,Y_train)


# In[125]:


bag_clf.score(X_test,Y_test)


# In[126]:


y_predict=bag_clf.predict(X_test)


# In[127]:


confusion_matrix(Y_test,y_predict)


# In[128]:


print(classification_report(Y_test,y_predict))


# In[129]:


precisions,recalls,thresholds=precision_recall_curve(Y_test,y_predict)


# In[130]:


plt.plot(precisions,recalls)


# In[131]:


fps,tps,thresholds=roc_curve(Y_test,y_predict)


# In[132]:


plt.plot(fps,tps)


# In[133]:


roc_auc_score(Y_test,y_predict)


# In[134]:


bag_clf=BaggingClassifier(svc,n_estimators=100,bootstrap=True)


# In[135]:


bag_clf.fit(X_train,Y_train)


# In[136]:


bag_clf.score(X_test,Y_test)


# In[137]:


y_predict=bag_clf.predict(X_test)


# In[138]:


confusion_matrix(Y_test,y_predict)


# # Adaboost Classifier

# In[254]:


from sklearn.ensemble import AdaBoostClassifier


# In[273]:


ada=AdaBoostClassifier(n_estimators=300)


# In[274]:


ada.fit(X_train,Y_train)


# In[275]:


ada.score(X_test,Y_test)


# In[276]:


y_predict=ada.predict(X_test)


# In[277]:


confusion_matrix(Y_test,y_predict)


# In[278]:


precision_score(Y_test,y_predict)


# In[279]:


recall_score(Y_test,y_predict)


# In[280]:


print(classification_report(Y_test,y_predict))


# # Testing on the Test Data for Test Score

# In[919]:


ada.score(x_test,y_test)


# In[920]:


y_predict=ada.predict(x_test)


# In[922]:


confusion_matrix(y_test,y_predict)


# In[923]:


precision_score(y_test,y_predict)


# In[924]:


recall_score(y_test,y_predict)


# In[925]:


print(classification_report(y_test,y_predict))


# In[926]:


precisions,recalls,thresholds=precision_recall_curve(y_test,y_predict)


# In[928]:


plt.plot(precisions,recalls)


# In[929]:


fps,tps,thresholds=roc_curve(y_test,y_predict)


# In[930]:


plt.plot(fps,tps)


# In[931]:


roc_auc_score(y_test,y_predict)


# # Gradient Boosting

# In[148]:


from sklearn.ensemble import GradientBoostingClassifier


# In[149]:


gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)


# In[150]:


gbc.fit(X_train,Y_train)


# In[151]:


gbc.score(X_test,Y_test)


# In[152]:


y_predict=gbc.predict(X_test)


# In[153]:


confusion_matrix(Y_test,y_predict)


# In[154]:


df_dropped


# # Feature Testing

# In[155]:


a=ada.feature_importances_


# In[156]:


t=pd.DataFrame(a)


# In[157]:


t["name"]=X_train.columns


# In[158]:


t.sort_values(by=0)


# # Performimg Cluster Analysis

# In[163]:


from sklearn.cluster import KMeans


# In[164]:


km=KMeans(2)


# In[165]:


km.fit(X)


# In[174]:


y=km.predict(X_test)


# In[177]:


km.inertia_


# In[178]:


confusion_matrix(Y_test,y)


# # Performing PCA on SEM 1 and SEM 2 grades

# In[214]:


pca=PCA(n_components=1)
p=pca.fit(X_train[["CORE_COURSE_GRADE_3_S","CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_2_S"]])
p=pca.transform(X_train[["CORE_COURSE_GRADE_3_S","CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_2_S"]])


# In[215]:


pca.explained_variance_ratio_


# In[216]:


pca.components_


# In[218]:


X_train["Sem_1_grades"]=p


# In[219]:


pca=PCA(n_components=1)
p1=pca.fit(X_train[["CORE_COURSE_GRADE_1_F","CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F","CORE_COURSE_GRADE_4_F"]])
p1=pca.transform(X_train[["CORE_COURSE_GRADE_1_F","CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F","CORE_COURSE_GRADE_4_F"]])


# In[220]:


pca.explained_variance_ratio_


# In[221]:


pca.components_


# In[222]:


X_train["Sem_2_grades"]=p1


# In[223]:


pca=PCA(n_components=1)
p=pca.fit(X_test[["CORE_COURSE_GRADE_3_S","CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_2_S"]])
p=pca.transform(X_test[["CORE_COURSE_GRADE_3_S","CORE_COURSE_GRADE_1_S","CORE_COURSE_GRADE_2_S"]])


# In[224]:


X_test["Sem_1_grades"]=p


# In[225]:


pca=PCA(n_components=1)
p1=pca.fit(X_test[["CORE_COURSE_GRADE_1_F","CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F","CORE_COURSE_GRADE_4_F"]])
p1=pca.transform(X_test[["CORE_COURSE_GRADE_1_F","CORE_COURSE_GRADE_2_F","CORE_COURSE_GRADE_3_F","CORE_COURSE_GRADE_4_F"]])


# In[226]:


X_test["Sem_2_grades"]=p1


# In[281]:


ada.fit(X_train,Y_train)


# In[282]:


ada.score(X_test,Y_test)


# In[283]:


y_predict=ada.predict(X_test)


# In[284]:


confusion_matrix(Y_test,y_predict)


# In[238]:


precision_score(Y_test,y_predict)


# In[239]:


recall_score(Y_test,y_predict)


# In[240]:


print(classification_report(Y_test,y_predict))


# # It is notable that after performing pca on the sem 1 and sem 2 grades the results don,t change 
