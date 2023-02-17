import os

######################################################################
### Change this path to the directory where you cloned the repo ######
repo_dir = "/home/ubuntu/hdd/COIGAN-controllable-object-inpainting"
######################################################################

model_path = os.path.join(repo_dir, "models_loss/ade20k/ade20k-resnet50dilated-ppm_deepsup")

# create the directory if it does not exist
os.makedirs(model_path, exist_ok=True)

# check if the system has wget installed
if os.system("which wget") != 0:
    raise Exception("Please install wget and run the script again.")

# download the model
os.system(f"wget -P {model_path} http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth")

# check if the model was downloaded correctly
if not os.path.exists(os.path.join(model_path, "encoder_epoch_20.pth")):
    raise Exception("The model was not downloaded correctly. Please try again.")

print("The model was downloaded successfully.")