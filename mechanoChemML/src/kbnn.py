import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout, Add
import keras.backend as K

def DNN(input_dim,hidden_units,output_dim=1,dropout=None,final_bias=False):
    # Define dense layers
    myLayers = []
    for i in range(len(hidden_units)):
        myLayers.append(Dense(hidden_units[i], activation='softplus',name=str(i)))
        if dropout:
           myLayers.append(Dropout(dropout))
    myLayers.append(Dense(output_dim,use_bias=final_bias,name='final'))
    
    # Define inputs and outputs
    inputs = Input(shape=(input_dim,))
    y = inputs
    for layer in myLayers:
        y = layer(y)
    outputs = y

    dnn = Model(inputs=inputs, outputs=outputs)
    return dnn

class KBNN():

    def __init__(self,LF,input_dim,hidden_units,output_dim=1,dropout=None,final_bias=False):
        # High fidelity "correcting" layers
        self.myLayers = []
        for i in range(len(hidden_units)):
            self.myLayers.append(Dense(hidden_units[i], activation='softplus'))
            if dropout:
                self.myLayers.append(Dropout(dropout))
        self.myLayers.append(Dense(output_dim))

        # Low fidelity model, lf = rho*LF(rho2*inputs+a2)
        # shift/scale layer, has weight of rho2 and bias of a2
        self.rho2_a2 = Dense(input_dim, kernel_initializer='ones')
        
        # Set LF model as not trainable
        self.LF = LF
        self.LF.trainable = False

        # weight LF contribution
        self.rho = Dense(output_dim, kernel_initializer='ones', use_bias=False)
        
        inputs = Input(shape=(input_dim,))
        # Pass input through HF correcting layers
        y = inputs
        for layer in self.myLayers:
            y = layer(y)
        correction = y

        # Pass input through shifted/scaled/weighted LF model
        # lf = self.rho*self.LF(self.rho2*inputs+self.a2)
        lf = inputs
        lf = self.rho2_a2(lf)
        lf = self.LF(lf)
        lf = self.rho(lf)

        # Combine in output layer
        outputs = Add()([correction, lf])

        self.model = Model(inputs=inputs,outputs=outputs)

        # self.LF = LF
        # self.rho = tf.Variable(1.)
        # self.rho2 = tf.Variable(np.ones(input_dim,dtype=np.float32))
        # self.a2 = tf.Variable(np.zeros(input_dim,dtype=np.float32))     

    def kbnn_loss(self, y_true, y_pred):
        c1 = 5.#10.
        c2 = 5.#10.
        c3 = 5.#10.
        MSE = keras.losses.mean_squared_error
        r = self.rho.get_weights()[0]
        r2 = self.rho2_a2.get_weights()[0]
        a2 = self.rho2_a2.get_weights()[1]
        #print('MSE(y_true, y_pred).shape: ', MSE(y_true,y_pred).shape)
        #print('a2.shape: ', a2.shape)
        return (MSE(y_true, y_pred) + c1*(r-1.)**2 + c2*tf.tensordot(r2-1,r2-1,axes=2) + c3*tf.tensordot(a2,a2,axes=1))
