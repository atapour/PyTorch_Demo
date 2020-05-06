# PyTorch Demonstration

This repository contains the code for a simple [PyTorch](https://pytorch.org/) demonstration. The code will train a three-layer MLP on the [MNIST](http://yann.lecun.com/exdb/mnist/) data.

## Get This Repository

Navigate to the directory where you want to place the code, and then run the following commands:

```
$ git clone https://github.com/atapour/PyTorch_Demo.git
$ cd PyTorch_Demo
```

You can run the code locally:

```
$ python pytorch_mlp.py
```

### Running the Code in Google Colaboratory

 - Navigate to - [https://colab.research.google.com](https://colab.research.google.com)
 - Sign in with Google account.

#### Using Google Colab Directly from Github
- Select File -> Upload Notebook... -> Github
- Paste this URL: [https://github.com/atapour/PyTorch_Demo/blob/master/PyTorch_MLP.ipynb](https://github.com/atapour/PyTorch_Demo/blob/master/PyTorch_MLP.ipynb)

#### Uploading the Notebook from the Local Copy

 - Select File -> Upload Notebook...
 - Drag and drop or browse to select the notebook you wish to use (`PyTorch_MLP.ipynb`).


 ### Important Note

 - As the code is specifically written to use a GPU, make sure you enable the use of a GPU in [Google Colab](https://colab.research.google.com).

 - Select Runtime -> Change runtime type -> GPU

 - Alternatively, you can change the first code cell of the notebook to use a CPU to run the code by including `device = torch.device('cpu')`.



