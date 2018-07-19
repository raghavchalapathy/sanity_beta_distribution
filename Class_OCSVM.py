class OCSVM:
    
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.metrics import roc_auc_score
    scaler = StandardScaler()

    def train_OCSVM_Classifier(self,X_train,nu,kernel):
        
        ocSVM = svm.OneClassSVM(nu = nu, kernel = kernel)
        ocSVM.fit(X_train) 

        return ocSVM
   
    def get_TestingData(self):

        dataPath = './'
        import tempfile
        import pickle

        with open(dataPath+'usps_data.pkl','rb') as fp:
              loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']
        
        ## Scale the data 
       
  
        print(scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        data = scaler.transform(data)
   
        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]


        data_ones = data_ones[220:440] 
        data_sevens = data_sevens =[0:11]
        # data_sevens =  np.random.uniform(0,1,(len(data_ones),256))
        
        label_ones      =  1*np.ones(len(data_ones))
        label_sevens    =  np.zeros(len(data_sevens))
        


        return [data_ones,label_ones,data_sevens,label_sevens]
    

        
        data_sevens =  np.random.uniform(0,1,(len(X),256))
        label_sevens    =  np.zeros(len(data_sevens))
        return [data_sevens,label_sevens]
  
    def get_TrainingData(self):
        
        dataPath = './'
        import tempfile
        import pickle
       

        with open(dataPath+'usps_data.pkl','rb') as fp:
              loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']
        
        ## Scale the data 
       
        print(scaler.fit(data))
        StandardScaler(copy=True, with_mean=True, with_std=True)
        data = scaler.transform(data)
   
        ## Select Ones
        k_ones = np.where(labels == 2)
        label_ones = labels[k_ones]
        data_ones = data[k_ones]

        k_sevens = np.where(labels == 8)
        label_sevens = labels[k_sevens]
        data_sevens = data[k_sevens]


        data_ones = data_ones[:220] 
        label_ones      =  1*np.ones(len(data_ones))
       
        return [data_ones,label_ones]
   
    def fit(self,X,nu,kernel):
  
        print("Training the OCSVM classifier.....")
        clf = self.train_OCSVM_Classifier(X,nu,kernel)

        return clf
   
    def compute_au_roc(self,y_true, df_score):
        y_scores_pos = df_score[0]
        y_scores_neg = df_score[1]
        y_score = np.concatenate((y_scores_pos, y_scores_neg))

        roc_score = roc_auc_score(y_true, y_score)
 
        return roc_score
          
    def predict(self,clf,Xtest_Pos,Xtest_Neg):
        decisionScore_POS = clf.decision_function(Xtest_Pos)
        decisionScore_Neg = clf.decision_function(Xtest_Neg)
        df_score = [ decisionScore_POS, decisionScore_Neg ]
        ## y_true
        y_true_pos = np.ones(data_test_normal.shape[0])
        y_true_neg = np.zeros(data_test_anomaly.shape[0])
        y_true = np.concatenate((y_true_pos, y_true_neg))

        plt.hist(decisionScore_POS, bins = 25, label = 'Normal')
        plt.hist(decisionScore_Neg, bins = 25, label = 'Anomaly')
        plt.legend(loc = 'upper right')
        plt.title('OC-SVM Normalised Decision Score')

        result = self.compute_au_roc(y_true,df_score)
        return result
        
        
ocsvm = OCSVM()
X_Pos,X_PosLabel = ocsvm.get_TrainingData()
[Xtest_Pos,label_ones,Xtest_Neg,label_sevens]= ocsvm.get_TestingData()
nu= 0.04
kernel = 'rbf'
clf = ocsvm.fit(X_Pos,nu,kernel)
res = ocsvm.predict(clf,Xtest_Pos,Xtest_Neg)
print("="*35)
print("AUC:",res)
print("="*35)