# Kaggle-Commpetition-Flower-Classification-TPUs-Petals-to-the-Metals
Welcome to the Petals to the Metal competition! In this competition, you’re challenged to build a machine learning model to classify 104 types of flowers based on their images. In this tutorial notebook, you'll learn how to build an image classifier in Keras and train it on a Tensor Processing Unit (TPU). At the end, you'll have a complete project you can build off of with ideas of your own.

To improve classification accuracy of the model on the test dataset, the following are explored:

1. Input image size
2. Pretrained model and number of trainable parameters of final model
3. Data augmentation
4. Regularization techniques
5. Use of learning rate schedule

### Step 0 : Import Libraries

we begin this notebook by importing useful analytics libraries, in which we import statistical, data visualization and milidating overfitting libraries along with tensorflow and keras.

### Step 1: Distribution Strategy

A TPU has eight different cores and each of these cores acts as its own accelerator. (A TPU is a sort of like having eight GPUs in one machine.) We tell TensorFlow how to make use of all these cores at once through a distribution strategy. Run the following cell to create the distribution strategy that we'll later apply to our model.

<b> What TPUClusterResolver() does? </b>

TPUs are network-connected accelerators and you must first locate them on the network. In TPUStrategy, the main objective is to contain the necessary distributed training code that will work on TPUs with their 8 compute cores. Whenever you use the TPUStrategy by instantiating your model in the scope of the strategy. This creates the model on the TPU. The model size is constrained by the TPU RAM only, not by the amount of memory available on the VM running your Python code. Model creation and model training use the usual Keras APIs. Further, read about [TPUClusterResolver()](https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/ClusterResolver) and [Kaggle TPU Doc](https://www.kaggle.com/docs/tpu)

We'll use the distribution strategy when we create our neural network model. Then, TensorFlow will distribute the training among the eight TPU cores by creating eight different replicas of the model, one for each core.

### Step 2: Loading The Competition Data

<b> Get GCS Path </b>

When used with TPUs, datasets need to be stored in a [Google Cloud Storage](https://cloud.google.com/storage/) bucket. You can use data from any public GCS bucket by giving its path just like you would data from '/kaggle/input'. The following will retrieve the GCS path for this competition's dataset.

You can use data from any public dataset here on Kaggle in just the same way. If you'd like to use data from one of your private datasets, see [here](https://www.kaggle.com/docs/tpu#tpu3pt5).

### Step 3: Loading Data (Setting up the parameters)

When used with TPUs, datasets are often serialized into TFRecords. This is a format convenient for distributing data to each of the TPUs cores. We've hidden the cell that reads the TFRecords for our dataset since the process is a bit long. You could come back to it later for some guidance on using your own datasets with TPUs.

TPU's is basically used to allocate the larger models having huge training inputs and batches, equipped with up to 128GB of high-speed memory allocation. In this notebook, we used an images dataset having pixel size is 512 x 512px, and see how TPU v3-8 handles it.

* num_parallel_reads=AUTO is used to automatically read multiple files.
* experimental_deterministic = False, we used "experimental_deterministic" to maintain the order of the data. Here, we disable the enforcement order to shuffle the data anyway.

<b> Tuning the Additional [Flower Data](https://www.kaggle.com/kirillblinov/tf-flower-photo-tfrec) </b>

To increase the proficiency of data, I have to use the external flower dataset with the helping material from [Dmitry's](https://www.kaggle.com/dmitrynokhrin/densenet201-aug-additional-data) and [Araik's](https://www.kaggle.com/atamazian/fc-ensemble-external-data-effnet-densenet) notebook. To visit the notebook to a better understanding of the Ensamble learning and augmentation of the external dataset.

<b> Data Augmentation </b>

This tutorial demonstrates data augmentation: a technique to increase the diversity of your training set by applying random (but realistic) transformations such as image rotation. [Read more](https://www.tensorflow.org/tutorials/images/data_augmentation)

TensorFlow Addons is a repository of contributions that conform to well-established API patterns but implement new functionality not available in core TensorFlow. TensorFlow natively supports a large number of operators, layers, metrics, losses, and optimizers. [Readout more](https://github.com/tensorflow/addons)

### Step4: Data Pipelines
### Step5: Data Exploration

Image Analysis with or without Augmentation

<b> Original versus w/ Random Augmentation </b>

![](/Images/1.png)

<b> UnBatch the Training Data </b>

![](/Images/2.png)

### Step6: Data Augmentation Sample

![](/Images/3.png)

### Data Augmentation Sample V2 (Implementing Image Processing)

![](/Images/4.png)

### Data Augmentation Sample V3 (Implementing Image Processing)

![](/Images/5.png)

### Step7: Defining The Model

Now we're ready to create a neural network for classifying images! We'll use what's known as transfer learning. With transfer learning, you reuse part of a pre-trained model to get a head-start on a new dataset.

For this tutorial, we'll use a model called VGG16 pre-trained on [ImageNet](https://image-net.org/)). Later, you might want to experiment with [other models](https://www.tensorflow.org/api_docs/python/tf/keras/applications) included with Keras. [Xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception) wouldn't be a bad choice.

The distribution strategy we created earlier contains a [context manager](https://docs.python.org/3/reference/compound_stmts.html#with), strategy.scope. This context manager tells TensorFlow how to divide the work of training among the eight TPU cores. When using TensorFlow with a TPU, it's important to define your model in a strategy.scope() context.

<b> To kepp track the model performance and findout the best suitable model through model-monitoring instance </b>

<b> Important : How to track learning rate during model training? </b>

Note: Stochastic gradient descent is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the back-propagation of errors algorithm, referred to as simply backpropagation. The amount that the weights are updated during training is referred to as the step size or the “learning rate.” Specifically, the learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value, often in the range between 0.0 and 1.0. For more information review the article of Jason Brownlee ["How to Configure the Learning Rate When Training Deep Learning Neural Networks"](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)

[Track learning rate during Training NotFoundError](https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard): Container worker does not exist. (Could not find resource: worker/_AnonymousVar8064) Encountered when executing an operation using EagerExecutor. This error cancels all future operations and poisons their output tensors.

![](https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard)

![](/Images/6.png)

Tuning Custom [Callbacks](https://www.tensorflow.org/guide/keras/custom_callback)

![](/Images/7.png)

Calculate the Weight of each [Flower Class](https://www.kaggle.com/xuanzhihuang/flower-classification-densenet-201)

![](/Images/8.png)

![](/Images/9.png)
