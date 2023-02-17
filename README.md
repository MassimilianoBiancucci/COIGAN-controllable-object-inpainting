<p>
<img align="left" width="120"  src="images/coilgun.png" style="margin-right:20px">
<h1> COIGAN controllable object inpainting</h1>
</p>
</br>
Project that aim to realize a GAN model capable of coherent objects inpainting inside a given image. This project took a lot from these two repositroies:

- [Swagan](https://github.com/rosinality/stylegan2-pytorch) A pytorch implementation of Stylegan2 based on wavelets, without progressive growing.
- [Lama](https://github.com/advimman/lama) A pytorch implementation of a GAN model for inpainting, that exploit the Fourier transform.

# Some results of the COIGAN model

TODO add here some graphs and images...

# Install the COIGAN module
To install the COIGAN module you need to clone the repo and install it with pip, so follow the commands listed below to install the module in your system:

```bash
cd /<project_target_path>/
git clone https://github.com/MassimilianoBiancucci/COIGAN-controllable-object-inpainting.git
cd COIGAN-controllable-object-inpainting
pip install -e COIGAN
```
Now the module is installed, follow the command bellow if you need to uninstall it:
```bash
pip uninstall COIGAN
```

# Prepare the Severstal steel defect dataset

## Setup the kaggle api
The first thing to do before the dataset preparation is to install and setup the kaggle api credentials, **if you already have the kaggle-api installed jump this step**.

install the the kaggle api in you system:
```bash
pip install kaggle
```
Now you need to retrive the username and the API-KEY and follow one of the following options:
- **options 1**: You can setup 2 env vars, one for the 
    ```bash
    export KAGGLE_USERNAME=datadinosaur
    export KAGGLE_KEY=xxxxxxxxxxxxxx
    ```
- **option 2**: You can setup a json file in the kaggle api files folder
    ```yaml
    /home/<username>/.kaggle/kaggle.json
    ```
    in the above file you need to add the user name and the API_KEY this way
    ```json
    {
        "username":"xxxxxxxxxxxxxx",
        "key":"xxxxxxxxxxxxxx"
    }
    ```
- **option 3**: You can download the zip file by yourself at the [the challenge page](https://www.kaggle.com/competitions/severstal-steel-defect-detection) and put it in the folder: 
    ```yaml
    /<your_path>/COIGAN-controllable-object-inpainting/data/severstal-steel-defect-detection
    ```

If you have any trouble with the kaggle api setup, refer to this link where the procedure is explained in detail: [Kaggle api repo](https://github.com/Kaggle/kaggle-api) or follow the option 3.

## Run the dataset preparation script
The first thing to do before launch the preparation script is to change the var **repo_dir** to the path of the repo in your local es: 

```yaml
repo_dir: /home/ubuntu/COIGAN-controllable-object-inpainting
```

the var is present inside the config file: 

```yaml
/home/ubuntu/COIGAN-controllable-object-inpainting/configs/data_preparation/severstal_dataset_preparation.yaml
```

### Data preparation settings
In this section are explained the principal variables of the dataset preparation script, to customize the dataset generated.
You can find those variables in the config file: 
```yaml
/your_path/COIGAN-controllable-object-inpainting/configs/data_preparation/severstal_dataset_preparation.yaml
```
Here the more interesting variables:

- **dataset_name**: name of the dataset, it will be used to create the folder where the generated dataset will be saved. The actual path will be `/<your_path>/COIGAN-controllable-object-inpainting/datasets/severstal-steel-defect-detection/<dataset_name>`
- **repo_dir**: path of the repo in your local, specify the main path of the COIGAN repo, es: `/<your_path>/COIGAN-controllable-object-inpainting`.
- **original_tile_size**: size of the original tile, for original tile is intended the dimension of the images in the input dtaset, for the severstal dataset is 1600x256, do not change this value.
- **tile_size**: size of the tile that will be generated, the tile will be cropped from the original tile, so the tile size must be smaller than the original tile size. This can be changed if you want a model with a different output shape.
- **binary**: This parameter specify if the datasets generated will be in binary json format, if set to True the read(+30%) and write(+150%) operations will be faster, and the dataset will be smaller in size.
- **n_workers**: number of workers used where the multithreading approach is used. -1 means that all the available cores will be used.
- **split_mode**: random or fair, specify the method used to split the samples in two sets, if **fair** is selected, the algorith try to create a train and test set with the same proportion of each class, and defected and non-defected samples. If **random** is selected, the samples are randomly splitted in two sets. 
- **train_ratio**: Specify how many samples will be placed in the train set from the original dataset, the value must be between 0 and 1.
- **test_ratio**: Specify how many samples will be placed in the test set from the original dataset, the value must be between 0 and 1.
- **black_threshold**: From the train dataset will extracted the samples without defects, to create a base dataset used as base for the inpainting process, but in the dataset there are images with big black areas, not suitable for the process, so this parameter specify the threshold used to segment the black area (consider 0 is totally black and 255 is totally white).
- **black_area_max_coverage**: Specify the maximum allowed extension of the black area on a sample, if greather than this value the sample won't be added to the base dataset.


# Train a COIGAN model

Before running the coigan training there are a few things to do, other than the dataset preparation. One step is to download the model used for the perceptual loss, and you need to review the training settings, changing the path of the COIGAN repo in the training configs.

## Download the needed models
For running the COIGAN training script and use the Resnet perceptual loss, you need to download the resnet50 model from the MIT Scene Parsing Benchmark dataset, you can download it running the script `download_models_loss.py` in the `COIGAN-controllable-object-inpainting/scripts` folder. NOTE: you should change the var `repo_dir` in the script to the path of the repo in your local es: 

```python
repo_dir= "/<your_path>/COIGAN-controllable-object-inpainting"
```

Then run the download script:

```bash
python download_models_loss.py
```
This will create a folder in the project root called `models_loss/ade20k/ade20k-resnet50dilated-ppm_deepsup` with inside the model checkpoint used for the loss. 