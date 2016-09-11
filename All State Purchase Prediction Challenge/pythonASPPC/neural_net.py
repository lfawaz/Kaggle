
from sknn.mlp import Classifier, Layer
from  sklearn.cross_validation import train_test_split
from utils import load_clean_data
import datetime
from sklearn import preprocessing

str(datetime.datetime.now())

#function to test mlp classifier
def test_mlp_classifier(data,layers=[Layer("Rectifier",units=10),Layer('Softmax')],learning_rate=0.02,n_iter=1,scale=1):
    
    #preprossing data if necessary 
    X_raw, y = data
    
    #normalize data for better performance
    if scale == 1:
        X = preprocessing.scale(X_raw)
    else:
        X = X_raw

    #since our test set is not labeled I am using the training data provided for train, validation and test
    print ("Create, train, test, validation_sets")
    
    #split the data into training/validation set and testing set
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X,y, test_size =0.2,random_state=42)
    
    #split the training set into training set and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,y_train_valid, test_size =0.2,random_state=23)
    
    #build the different layers of the model
    print("Building the model...")
    nn = Classifier(
        layers=layers,
        learning_rate=learning_rate,
        n_iter=n_iter)
    
    #train the model
    print("Training...")
    nn.fit(X_train, y_train)
    
    #test the model
    print("Testing...")
    y_valid = nn.predict(X_train)
    
    #return the validation score
    print("Score...")
    score = nn.score(X_test, y_test)
    
    
    return score,layers,learning_rate,n_iter


def main():
    
    #load the data
    data = load_clean_data()
    
    #create a list with all the configuration that we wanted to run 
    #This is created because each configuration run for over an hour
    network_configs = [{"layers":"[Layer('Rectifier',units=20,weight_decay=0.00001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.02,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=40,weight_decay=0.0001,dropout=0.01),Layer('Softmax')]","learning_rate": 0.03,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=60,weight_decay=0.001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=80,weight_decay=0.0001,dropout=0.4),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=100,weight_decay=0.1,dropout=0.001),Layer('Softmax')]","learning_rate": 0.001,"n_inter":150},
              {"layers":"[Layer('Rectifier',units=20,weight_decay=0.00001,dropout=0.1),Layer('Rectifier',units=20,weight_decay=0.0001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=40,weight_decay=0.001,dropout=0.01),Layer('Rectifier',units=40,weight_decay=0.0001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=60,weight_decay=0.0001,dropout=0.4),Layer('Rectifier',units=60,weight_decay=0.0001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=80,weight_decay=0.1,dropout=0.1),Layer('Rectifier',units=80,weight_decay=0.0001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 150},
              {"layers":"[Layer('Rectifier',units=100,weight_decay=0.0001,dropout=0.001),Layer('Rectifier',units=100,weight_decay=0.0001,dropout=0.1),Layer('Softmax')]","learning_rate": 0.001,"n_inter": 150}]

#This loop will run each configration in the list above
#and then write the results to the flat file 'neural net score'

    for config in network_configs:       
        f = open('neural_net_score','a')

        start = "\n start:  %s | " %(str(datetime.datetime.now()))
        print('start time....'), start
        
        f.write(start)
        
        layers = config.get('layers')
        learning_rate = config.get('learning_rate')
        n_iter = config.get('n_inter')
        print "Configs...",layers,
        score,layers,learning_rate,n_iter = test_mlp_classifier(data,layers=eval(layers),learning_rate=learning_rate,n_iter=n_iter,scale=1)
        line = ("score: %s, layers: %s, learning_rate: %s, n_iter %s" %(score, layers, learning_rate, n_iter))
        
        f.write(line)
        End = " | End %s" %(str(datetime.datetime.now()))
        f.write(End)
        f.close()


if __name__ == "__main__":
    main()


# In[ ]:



