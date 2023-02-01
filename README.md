<p>
<img align="left" width="120"  src="images/coilgun.png" style="margin-right:20px">
<h1> COIGAN controllable object inpainting</h1>
</p>

Project that aim to realize a GAN model capable of objects generation inside a given image, with structure based on stylegan2 and lama.

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
In this section are explained all the variables of the dataset preparation script, to customize the dataset generated.


