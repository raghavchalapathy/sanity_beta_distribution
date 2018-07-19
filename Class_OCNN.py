class OCNN:
    
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.metrics import roc_auc_score
    import tensorflow as tf
    import numpy as np
    import numpy  as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as srn
  
    

    results = "./sanity_results/"
    decision_scorePath = "./scores/"
    df_usps_scores  = {}
    activations = ["Linear","Sigmoid"]
    methods = ["Linear","RBF"]
    path = "./scores/"
   
    nu = 0.1
    scaler = StandardScaler()

    

    def write_Scores2Csv(self,train, trainscore, test, testscore,filename):

            data = np.concatenate((train, test))
            score= np.concatenate((trainscore,testscore))
            data = data.tolist()
            score = score.tolist()
            with open(filename, 'a') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(("x", "score"))
            for row in range(0,len(data)):
                with open(filename,
                        'a') as myfile:
                    wr = csv.writer(myfile)

                    wr.writerow((" ".join(str(x) for x in data[row]), " ".join(str(x) for x in score[row])))
    def write_decisionScores2Csv(self,path, filename, positiveScores, negativeScores):

            newfilePath = path+filename
            print("Writing file to ", path+filename)
            poslist = positiveScores.tolist()
            neglist = negativeScores.tolist()

            # rows = zip(poslist, neglist)
            d = [poslist, neglist]
            export_data = izip_longest(*d, fillvalue='')
            with open(newfilePath, 'w') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(("Normal", "Anomaly"))
                wr.writerows(export_data)
            myfile.close()

            return

    def train_OCNN_Classifier(self,X_train,nu,activation,epochs):

        RANDOM_SEED = 42
        tf.reset_default_graph()
        train_X = X_train
        tf.set_random_seed(RANDOM_SEED)
        outfile = "./model_weights/"
        oCSVMweights = "./weights/"
        import time

        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
        h_size = 200                # Number of hidden nodes
        y_size = 1   # Number of outcomes (3 iris flowers)
        D = x_size
        K = h_size
        theta = np.random.normal(0, 1, K + K*D + 1)
        rvalue = np.random.normal(0,1,(len(train_X),y_size))
        g   = lambda x : (1/np.sqrt(h_size) )*tf.cos(x/0.02)

        def init_weights(shape):
            """ Weight initialization """
            weights = tf.random_normal(shape,mean=0, stddev=0.5)
            return tf.Variable(weights,trainable=False)

            def forwardprop(X, w_1, w_2):
                """
                Forward-propagation.
                IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
                """
                X = tf.cast(X, tf.float32)
                w_1 = tf.cast(w_1, tf.float32)
                w_2 = tf.cast(w_2, tf.float32)
                h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
                yhat = tf.matmul(h, w_2)  # The \varphi function
                return yhat
        
      
        
        def nnScore(X, w, V, g,bias1,bias2):
            X = tf.cast(X, tf.float32)
            w = tf.cast(w, tf.float32)
            V = tf.cast(V, tf.float32)
            y_hat =tf.matmul(g((tf.matmul(X, w)+bias1)), V) +bias2

            return y_hat
        
        def relu(x):
            y = x
            y[y < 0] = 0
            return y
        
        # For testing the algorithm
        def compute_LossValue(X, nu, w1, w2, g, r,bias1,bias2):
            w = w1
            V = w2

            X = tf.cast(X, tf.float32)
            w = tf.cast(w1, tf.float32)
            V = tf.cast(w2, tf.float32)
            term1 = 0.5 * tf.reduce_sum(tf.square(w))
            term2 = 0.5 * tf.reduce_sum(tf.square(V))


            
            term3 = 1 / nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g,bias1,bias2)))
            term4 = -r
            
            y_hat = nnScore(X, w, V, g,bias1,bias2)
            
            totalCost = term1 + term2 + term3 + term4
                
            loss=   [term1,term2,term3,term4,totalCost,y_hat]
            
            return loss
            
            
        def ocnn_obj(theta, X, nu, w1, w2, g,r,bias1,bias2):

            w = w1
            V = w2
     
            X = tf.cast(X, tf.float32)
            w = tf.cast(w1, tf.float32)
            V = tf.cast(w2, tf.float32)


            term1 = 0.5  * tf.reduce_sum(w**2)
            term2 = 0.5  * tf.reduce_sum(V**2)
            term3 = 1/nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g,bias1,bias2)))
            term4 = -r

            return term1 + term2 + term3 + term4





            # Symbols
        X = tf.placeholder("float32", shape=[None, x_size])

        r = tf.get_variable("r", dtype=tf.float32,shape=())

        # Weight initializations
        w_1 = init_weights((x_size, h_size))
           
        weights = tf.random_normal((h_size, y_size),mean=0, stddev=0.1)
           
        ocsvm_wt = np.load(oCSVMweights+"ocsvm_wt.npy")
        w_2 =tf.get_variable("tf_var_initialized_ocsvm",
                                initializer=ocsvm_wt)
            
        bias1 = tf.Variable(initial_value=[[1.0]], dtype=tf.float32,trainable=False)
        bias2 = tf.Variable(initial_value=[[0.0]], dtype=tf.float32,trainable=False)


        cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r,bias1,bias2)
        #updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        updates = tf.train.AdamOptimizer(4.7 * 1e-1).minimize(cost)

        # Run SGD
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        rvalue = 0.1
        start_time = time.time()
        print("Training OC-NN started for epochs: ",epochs)
        for epoch in range(epochs):
                    # Train with each example
            sess.run(updates, feed_dict={X: train_X})
                        
                    
            with sess.as_default():
                svalue = nnScore(train_X, w_1, w_2, g,bias1,bias2)  
                print ("Checking svalue <rvalue: ",np.mean(svalue.eval() < rvalue))
                rval = svalue.eval()
                rvalue = np.percentile(rval,q=100*nu)
                            
                print ("Checking svalue <rvalue: ",np.mean(svalue.eval() < rvalue))
                costvalue = compute_LossValue(train_X, nu, w_1, w_2, g, rvalue,bias1,bias2)
                term1 = costvalue[0].eval()
                term2 = costvalue[1].eval()
                term3 = costvalue[2].eval()
                term4 = costvalue[3]
                term5 = costvalue[4].eval()
                yval = costvalue[5].eval()
        
                print("Epoch = %d, r = %f"
                        % (epoch + 1,rvalue))
                print ("Cost:(term1,term2,term3,term4,term5,yhat) ", np.mean(term1),np.mean(term2),(term3),np.mean(term4),np.mean(term5),np.mean(yval))
                print ("Total Cost: ",np.mean(term5))
                        
                with sess.as_default():
                    print ("Checking svalue <rvalue: ",np.mean(svalue.eval() < rvalue))
                    print ("================================")
                    print ("================================")

            import time
            trainTime = time.time() - start_time
            print("Training Time taken,",trainTime)
          
            
            
            with sess.as_default():
                np_w_1= w_1.eval()
                np_w_2= w_2.eval()
                np_bias1= bias1.eval()
                np_bias2= bias2.eval()
            
            rstar =rvalue
#             sess.close()
#             print("Session Closed!!!")

            # save the w_1 and bias1 to numpy array
            print("Saving the trained Model weights ... @",outfile)
            print("The optimized value of r found is",rstar)
            np.save(outfile+"w_1", np_w_1)
            np.save(outfile+"w_2", np_w_2)
            np.save(outfile+"bias1",np_bias1)
            np.save(outfile+"bias2",np_bias2)

   
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
        data_sevens = data_sevens[0:11]
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
     
    def fit(self,X,nu,activation,epochs):
  
        print("Training the OCNN classifier.....")
        self.train_OCNN_Classifier(X,nu,activation,epochs)

        return   
    
    def compute_au_roc(self,y_true, df_score):
        y_scores_pos = df_score[0]
        y_scores_neg = df_score[1]
        y_score = np.concatenate((y_scores_pos, y_scores_neg))

        roc_score = roc_auc_score(y_true, y_score)
 
        return roc_score
    
    def decision_function(self,X, w_1, w_2, g,bias1,bias2):   
        score =np.matmul(g((np.matmul(X, w)+bias1)), V) +bias2
        return score

    def predict(self,Xtest_Pos,Xtest_Neg):
        
        ## Load the saved model and compute the decision score
        w_1 = np.load(model_weights+"/w_1.npy")
        w_2 = np.load(model_weights+"/w_2.npy")
        bias1 = np.load(model_weights+"/bias1.npy")
        bias2 = np.load(model_weights+"/bias2.npy")

        decisionScore_POS= decision_function(Xtest_Pos, w_1, w_2, g,bias1,bias2)
        decisionScore_Neg = decision_function(Xtest_Neg, w_1, w_2, g,bias1,bias2)
   
        df_score = [decisionScore_POS, decisionScore_Neg]
        
        ## y_true
        y_true_pos = np.ones(data_test_normal.shape[0])
        y_true_neg = np.zeros(data_test_anomaly.shape[0])
        y_true = np.concatenate((y_true_pos, y_true_neg))

        plt.hist(decisionScore_POS, bins = 25, label = 'Normal')
        plt.hist(decisionScore_Neg, bins = 25, label = 'Anomaly')
        plt.legend(loc = 'upper right')
        plt.title('OC-NN Normalised Decision Score')

        result = self.compute_au_roc(y_true,df_score)
        return result
        


## Instantiate the object and call the function
ocnn = OCNN()
X_Pos,X_PosLabel = ocnn.get_TrainingData()
[Xtest_Pos,label_ones,Xtest_Neg,label_sevens]= ocnn.get_TestingData()
nu= 0.04
activation = 'sigmoid'
epochs = 10
ocnn.fit(X_Pos,nu,activation,epochs)
res = ocnn.predict(Xtest_Pos,Xtest_Neg)
print("="*35)
print("AUC:",res)
print("="*35)




