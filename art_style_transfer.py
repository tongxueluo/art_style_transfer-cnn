'''A tensorflow implementation of A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576). Using vgg-19.'''

import tensorflow as tf
import numpy as np
import scipy
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import os

class art_style_transfer:

    def __init__(self, noise_ratio = 0.6, beta = 5, alpha = 100, PRINTOUT=True,
                 optimizer = tf.train.AdadeltaOptimizer(2.0), iterations = 5000):
        # Algorithm constants:
        self.noise_ratio = noise_ratio #percentage of weight of the noise for intermixing with the content image
        self.beta = beta #constant to put more emphasis on content loss
        self.alpha = alpha
        self.PRINTOUT = PRINTOUT
        self.optimizer = optimizer
        self.iterations = iterations

        # image dimensions constants
        self.img_wth = 196
        self.img_hgt = 256
        self.clr_cnl = 3
        # vgg19 for image recognition
        self.vgg_model = "./imagenet-vgg-verydeep-19.mat"
        self.mean_values = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

        self.style_layers = [('conv1_1',0.5),('conv2_1',1.0),('conv3_1',1.5),('conv4_1',3.0),('conv5_1',4.0)]

    def __load_vgg_model(self):
        # Takes only the convolution layer weights and wrap using the TensorFlow Conv2d, Relu and AveragePooling layer.
        # VGG actually uses maxpool but the paper indicates that using AveragePooling yields better results.
        # The last few fully connected layers are not used.
        vgg_model = self.vgg_model
        vgg = scipy.io.loadmat(vgg_model)
        vgg_layers = vgg['layers']
        print "vgg_layers.shape: ", vgg_layers.shape
        def _weights(layer, expected_layer_name):
            # return the weights and bias from the vgg for a given layer
            W = vgg_layers[0][layer][0][0][2][0][0]
            b = vgg_layers[0][layer][0][0][2][0][1]
            layer_name = vgg_layers[0][layer][0][0][0]
            print("layer_name = ", layer_name, "expected = ",expected_layer_name)
            assert layer_name == expected_layer_name
            return W,b
        def _relu(conv2d_layer):
            # return the RELU function wrapped over a TensorFlow layer. Expects a Conv2d layer input.
            return tf.nn.relu(conv2d_layer)
        def _conv2d(prev_layer, layer, layer_name):
            # return the conv2d layer using the weights, biases from the vgg model at 'layer'
            W, b = _weights(layer, layer_name)
            W = tf.constant(W)
            b = tf.constant(np.reshape(b, (np.array(b).size)))
            print "W_shape, b_shape: ", W.shape, b.shape
            test = tf.nn.conv2d(prev_layer, filter=W, strides=[1,1,1,1], padding='SAME')
            print 'test', test.shape
            return tf.nn.conv2d(prev_layer, filter=W, strides=[1,1,1,1], padding='SAME')+b
        def _conv2d_relu(prev_layer, layer, layer_name):
            # return the conv2d+relu layer using the weights, biases from the vgg model at 'layer'
            return _relu(_conv2d(prev_layer,layer,layer_name))
        def _avgpool(prev_layer):
            #return the AveragePooling layer
            return tf.nn.avg_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #constructs the graph model
        graph = {}
        graph['input'] = tf.Variable(np.zeros((1,self.img_hgt,self.img_wth,self.clr_cnl)), dtype='float32')
        graph['conv1_1'] = _conv2d_relu(graph['input'],0,'conv1_1')
        graph['conv1_2'] = _conv2d_relu(graph['conv1_1'],2,'conv1_2')
        graph['avgpool1'] = _avgpool(graph['conv1_2'])
        graph['conv2_1']  = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2']  = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = _avgpool(graph['conv2_2'])
        graph['conv3_1']  = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2']  = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3']  = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4']  = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = _avgpool(graph['conv3_4'])
        graph['conv4_1']  = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2']  = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3']  = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4']  = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = _avgpool(graph['conv4_4'])
        graph['conv5_1']  = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2']  = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3']  = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4']  = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')  #34 is conv5_4 (3, 3, 512, 512)
        graph['avgpool5'] = _avgpool(graph['conv5_4'])
        return graph

    def __content_loss_func(self, sess, model):
        # content loss function as defined in the paper. Here only concerned with the 'conv4_2' layer
        def _content_loss(p,x):
            N = p.shape[3]  #N is the number of filters (at layer l)
            M = p.shape[1]*p.shape[2] #M is the height times the width of the feature map (at layer l)
            return (1.0/float(4*N*M))*tf.reduce_sum(tf.pow(x-p,2))
        return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

    def __style_loss_func(self, sess, model):
        # style loss function as defined in the paper
        def _gram_matrix(F,N,M):  #the gram matrix G.
            Ft = tf.reshape(F, (M,N))
            return tf.matmul(tf.transpose(Ft),Ft)
        def _style_loss(a,x):
            N = a.shape[3] #N is the number of filters (at layer l)
            M = a.shape[1]*a.shape[2] #M is the height times the width of the feature map (at layer l)
            A = _gram_matrix(a,N,M)
            G = _gram_matrix(x,N,M)
            result = (1.0/float(4*N**2*M**2))*tf.reduce_sum(tf.pow(G-A,2))
            return result
        E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in self.style_layers]
        W = [w for _, w in self.style_layers]
        loss = sum([W[l]*E[l] for l in range(len(self.style_layers))])
        return loss

    def __generate_noise_image(self, content_image):
        # return a noise image intermixed with the content image at a certain ratio
        noise_ratio = self.noise_ratio
        noise_image = np.random.uniform(-20,20, (1, self.img_hgt, self.img_wth, self.clr_cnl)).astype('float32')
        input_image = noise_image*noise_ratio + content_image*(1-noise_ratio)
        if self.PRINTOUT:
            plt.imshow(input_image[0])
        return input_image

    def __load_image(self, path):
        image = Image.open(path)
        image = image.resize((self.img_wth, self.img_hgt))  #resize the image
        image = np.array(image)
        image = np.reshape(image, ((1,)+image.shape)) #add an extra dimension
        image = image - self.mean_values
        return image
    def __save_image(self, path, image):
        image = image+self.mean_values #output should add back the mean
        image = image[0]               #get rid of the first useless dimension
        image = np.clip(image,0,255).astype('uint64')
        scipy.misc.imsave(path,image)

    def generate_art(self, content_image, style_image):
        sess = tf.InteractiveSession()
        output_dir = "./output_dir"
        if os.path.exists(output_dir) and len(os.listdir(output_dir)[1:])>0:
            content_image = self.__load_image(content_image)
            file = os.listdir(output_dir)[1:]
            def tryint(s):
                try:
                  return int(s)
                except:
                  return s
            import re
            def alphanum_key(s):
                return [ tryint(c) for c in re.split('([0-9]+)', s) ]
            file.sort(key= alphanum_key)
            saved_img = file[-1]
            for i in alphanum_key(saved_img):
                if isinstance( i, int ):
                    number_to_continue = i
            input_image = self.__load_image(output_dir +"/" + saved_img)
            print "%s as the input_image."% saved_img
        else:
            content_image = self.__load_image(content_image)
            input_image = self.__generate_noise_image(content_image)
            number_to_continue = 0
        style_image = self.__load_image(style_image)
        model = self.__load_vgg_model()
        if self.PRINTOUT:
            print "****************check images***********"
            print "content and style and input imgaes shapes: ", content_image.shape, style_image.shape, input_image.shape

        sess.run(tf.global_variables_initializer())
        # construct content_loss using content_image
        print "**************check point 1********"
        sess.run(model['input'].assign(content_image))
        content_loss = self.__content_loss_func(sess,model)
        print "content loss: ", sess.run(content_loss)
        # construct style_loss using style_image
        print "**************check point 2********"
        sess.run(model['input'].assign(style_image))
        style_loss = self.__style_loss_func(sess,model)
        print "style loss: ", sess.run(style_loss)
        print "**************check point 3********"
        # instantiate equation 7 of the paper
        total_loss = self.beta*content_loss + self.alpha*style_loss
        print "total loss: ", total_loss, sess.run(total_loss)

        # "jointly minimize the distance of a white noise image
        # from the content representation of the photograph in one layer of
        # the network and the style representation of the painting in a number
        # of layers of the CNN
        #
        # the content is built from on layer, while the style is from five
        # layers. Then minimize the total_loss, which is the equation 7.

        train_step = self.optimizer.minimize(total_loss)

        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))
        print("******************************")
        img1 = sess.run(model['conv1_1'])
        img2 = sess.run(model['input'])
        print input_image.shape, img1.shape, img2.shape
        print('sum: ', sess.run(tf.reduce_sum(input_image-img2)))
        print('cost: ', sess.run(total_loss))
        print('sum: ', sess.run(tf.reduce_sum(img1)))
        print('cost: ', sess.run(total_loss))
        print('sum: ', sess.run(tf.reduce_sum(img2)))
        print('cost: ', sess.run(total_loss))

        print "**************check point 4********"
        for it in range(self.iterations):
            sess.run(train_step)
            if it%100 == 0:
                # print every 100 iteration
                mixed_image = sess.run(model['input'])
                print('Iteration %d'%(it))
                print('sum: ', sess.run(tf.reduce_sum(mixed_image)))
                print('cost: ', sess.run(total_loss))
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                filename = '/output %d.png' % (it+number_to_continue)
                self.__save_image(output_dir + filename, mixed_image)

if __name__ == '__main__':
    art_transfer = art_style_transfer(noise_ratio=0.1)
    content_img = "./madonna_del_prato.jpg"
    style_img = "./hanyatu.jpg"
    newart = art_transfer.generate_art(content_image=content_img, style_image=style_img)







