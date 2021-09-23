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




