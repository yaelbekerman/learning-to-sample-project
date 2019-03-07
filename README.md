# Learning to sample variations project

Variations on S-NET from "learning to sample" - https://github.com/orendv/learning_to_sample


### Installation

The original code was tested on Python 2.7.12, TensorFlow 1.2.1, CUDA 8.0 and cuDNN 5.1.1 on Ubuntu 16.04.
We tested our new variaitons on Python 2.7.15, Tensorflow 1.12.0, CUDA 9.0 and cuDNN 7.2.1 on Ubuntu 16.04 and on the same setup but with CUDA 9.2. CUDA 9.0 worked better.
The original setup tested by the original code should probably work as well.


You may need to install h5py and wget.


Compile the structural losses using the make file:

```
cd structural_losses/
make
```

As explained in the original github, you might need to change the first few lines of the make file to point to your nvcc, tensorflow and cuda libraries.



### Usage

#### PointNet training

To train PointNet run the following command providing the saving directory:

```
python train_classifier.py --log_dir log/baseline/PointNet1024
```

#### Unsupervised S-NET

To train S-NET with unlabled data run the following command providing the saved PointNet model, number of points to sample and log directory:

```
python train_SNET_unsupervised.py --classifier_model_path log/baseline/PointNet1024/model.ckpt --num_out_points 64 --log_dir log/SNET64UNSUPERVISED
```

To evaluate run:

```
python evaluate_SNET_unsupervised.py --sampler_model_path log/SNET64UNSUPERVISED/model.ckpt --dump_dir log/SNET64UNSUPERVISED/eval --num_out_points 64
```

#### Unsupervised S-NET with threashold

To use a threashold when training, run the following command providing the previous arguments as well as the threashold value

```
python train_SNET_unsupervised_threashold.py --unsupervised_threashold 0.8 --classifier_model_path log/baseline/PointNet1024/model.ckpt --num_out_points 64 --log_dir log/SNET64UNSUPERVISEDTH8
```

To evaluate run:

```
python evaluate_SNET_unsupervised.py --sampler_model_path log/SNET64UNSUPERVISEDTH8/model.ckpt --dump_dir log/SNET64UNSUPERVISEDTH8/eval --num_out_points 64
```


#### S-NET with smaller dataset

To train S-NET on a smaller dataset 

```
bla bla bla....a.a.a.a.a.a.a..askalsjchsliduhcldwjncoidwq ncqihdgcwqlkjbc iuwqkdcboiuewqbd iew hbdckuhqbdc kuwqhbckuwqhbdckquwhbd
```

### Acknowledgment

Our project and code are based on the work by Oren Dovrat et al. available at https://github.com/orendv/learning_to_sample.
We want to thank the authors for publishing their code.

We also want to thank on the work by Qi et al. available at https://github.com/charlesq34/pointnet for publishing their code.

