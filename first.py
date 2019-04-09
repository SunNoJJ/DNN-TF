#!/home/user/anaconda3/bin/python3 
#-*-coding:utf-8-*-
##  practice.py
##  retract = 4space

import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

model_path = "model"
model_name = "GG"
max_epoch = 50## 最大训练次数
## 创建保存模型的文件夹
if not os.path.exists(model_path):
    os.mkdir(model_path)
## 产生的随机数 进行模型测试
X = np.random.normal(0.0,1.0,(400,100))
Y = np.random.normal(0.0,1.0,(400,50))


class dnnnet():
    def __init__(self,x):
        self.x = x
        pass
    
    def add_layer(self,input_x,input_size,output_size,activation_function = None):
        weights = tf.Variable(tf.random_normal([input_size,output_size]),dtype = tf.float32)
        biases = tf.Variable(tf.random_normal([1,output_size]),dtype = tf.float32)
        xw = tf.matmul(input_x,weights)
        xw_plus_b = tf.add(xw,biases)
        if activation_function == None:
            return xw_plus_b
        else:
            return activation_function(xw_plus_b)
    
    def net_structure(self):
        layer_1 = self.add_layer(self.x,100,200,tf.nn.relu)## input_size 100;L1 = 200
        layer_2 = self.add_layer(layer_1,200,300,tf.nn.sigmoid)## L2 = 300
        layer_3 = self.add_layer(layer_2,300,50,tf.nn.tanh)## L3 = 50
        return layer_3
        
def train():
    ## 输入占位符 不能作为全局变量 否则、
    ## You must feed a value for placeholder tensor 'input_1' with dtype
    X1 = tf.placeholder(tf.float32,[None,100],name = 'input')
    Y1 = tf.placeholder(tf.float32,[None,50])
    ## train  训练 
    net = dnnnet(X1)
    y_pre = tf.nn.softmax(net.net_structure(),name = 'output')      
    ##  选择损失函数计算  残差
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pre , labels = Y1))                        
    ## 选择优化器  设置优化学习率
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.025)
    ## 设置优化目标
    train = optimizer.minimize(loss)## train
    my_global_step = tf.Variable(0, trainable=True,name = 'my_global_step') 
    ## 初始化全部Tensor
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep = 2)
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state('./'+model_path)
        if ckpt is not None:## 导入模型进行增量训练
            path = ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....'% path)
            saver.restore(sess,path)
        epoch = sess.run(my_global_step)
        print("epoch:{:5d}".format(epoch))
        mini_loss = 999. ## 最小残差 
        while True:
            if mini_loss<0.005 or epoch > max_epoch:break
            _, cost_output = sess.run([train, loss],feed_dict = {X1:X,Y1:Y})## train some 
            ### 保存最好模型
            if cost_output < mini_loss:
                sess.run(tf.assign(my_global_step, epoch))
                mini_loss = cost_output
                saver.save(sess, './'+model_path+'/'+model_name,global_step = epoch)
            print("epoch:{:5d} ||loss:{:.5f} ||miniloss:{:.5f}".format(epoch,cost_output,mini_loss))
            epoch += 1
        print("mini_loss:{:.4f}".format(mini_loss))
## ~~~~~~~~ test ~~~~~~~~ ##
def test():
    class_num = 50
    x_batch = X[0:50,:]
    y_batch = Y[0:50,:]
 
    test_batch = len(x_batch)
    xxx_plt = range(test_batch)    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    real_lable = np.dot(y_batch ,np.ones((class_num,1)))#[[0.],[1.]]
    ax.scatter(xxx_plt,real_lable,c = 'r',marker = '*')
    plt.ion()### continiu
    plt.show()

    #获取最新路径  
    checkpoint = tf.train.get_checkpoint_state("./"+model_path)    
    model_name = checkpoint.model_checkpoint_path+".meta"
    #打开会话
    with tf.Session() as sess:		
        print("===================",model_name)
        #载入模型  
        saver = tf.train.import_meta_graph(model_name)
	    #恢复模型
        saver.restore(sess,tf.train.latest_checkpoint('./'+model_path+'/'))          
	    #使用模型进行预测
        pred = sess.run('output:0',feed_dict = {"input:0":x_batch})
        pre = np.dot(pred ,np.array(range(0,class_num)).reshape(class_num,1))
        rightNum = 0  
        for i, preid in enumerate(pre):
            ax.plot(i, preid, 'co', lw = 0.5)                   
            plt.pause(0.1)            

        title = 'right_rate:'+'{:.2f}'.format(rightNum/test_batch*100.0)+'%'
        plt.title(title)    
        plt.savefig('predict.png')
    print("GG")
    
    
    
if __name__ == "__main__":
    if sys.argv[1]=="test":test()       
    if sys.argv[1]=="train":train()
        
        



