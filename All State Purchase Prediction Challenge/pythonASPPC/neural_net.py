
# coding: utf-8

# In[16]:

from sknn.mlp import Classifier, Layer
from  sklearn.cross_validation import train_test_split
from utils import grid_search,load_clean_data
import datetime

str(datetime.datetime.now())
def test_mlp_classifier(X,y,layers=[Layer("Rectifier",units=10),Layer('Softmax')],learning_rate=0.02,n_iter=1):
    
   
    #since our test set is not labeled I am using the training data provided for train, validation and test
    print ("Create, train, test, validation_sets")
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X,y, test_size =0.2,random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,y_train_valid, test_size =0.2,random_state=23)
    
    print("Building the model...")
    nn = Classifier(
        layers=layers,
        learning_rate=learning_rate,
        n_iter=n_iter)
    
    print("Training...")
    nn.fit(X_train, y_train)
    
    print("Testing...")
    y_valid = nn.predict(X_train)

    print("Score...")
    score = nn.score(X_test, y_test)
    
    return score,layers,learning_rate,n_iter


# In[18]:

def main():
    X,y = load_clean_data()
    
    network_configs = [{"layers":"[Layer('Rectifier',units=20),Layer('Softmax')]","learning_rate": 0.02,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=40),Layer('Softmax')]","learning_rate": 0.03,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=60),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=80),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=100),Layer('Softmax')]","learning_rate": 0.001,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=20),Layer('Rectifier',units=20),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=40),Layer('Rectifier',units=40),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=60),Layer('Rectifier',units=60),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=80),Layer('Rectifier',units=80),Layer('Softmax')]","learning_rate": 0.01,"n_inter": 100},
              {"layers":"[Layer('Rectifier',units=100),Layer('Rectifier',units=100),Layer('Softmax')]","learning_rate": 0.001,"n_inter": 100}]
    
    for config in network_configs:       
        score,layers,learning_rate,n_iter = test_mlp_classifier(X=X,y=y,layers=eval(config.get('layers')),learning_rate=config.get('learning_rate'),n_iter=config.get('n_inter'))
        line = ("score: %s, layers: %s, learning_rate: %s, n_iter %s" %(score, layers, learning_rate, n_iter))
        f = open('neural_net_score','a')
        f.write("\n start:  %s | " %(str(datetime.datetime.now())))
        f.write(line)
        f.write(" | End %s" %(str(datetime.datetime.now())))
        f.close()


if __name__ == "__main__":
    main()


# In[ ]:




# In[ ]:



