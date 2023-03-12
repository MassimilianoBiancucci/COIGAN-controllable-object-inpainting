import os
import logging

import cv2
import numpy as np

from omegaconf import DictConfig

from COIGAN.inference.coigan_inference import COIGANinference
from COIGAN.inference.mask_gui_window import MaskGuiWindow

LOGGER = logging.getLogger(__name__)

class COIGANinferenceGui:

    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Inti method of the COIGAN inference GUI.
        this metho initialize a COIGANinference object and then
        many cv2 windows to visualize and interact with the model.
        The interaction is mainly done with the mouse, drawing the masks
        of each class.

        Args:
            config (DictConfig): config of the model
        """

        # save the config
        self.config = config

        # load the classes
        self.classes = self.config.classes

        # create the inference object
        LOGGER.info("Creating the COIGAN inference model")
        self.model = COIGANinference(config)

        # loading the input images
        LOGGER.info("Loading the input images")
        self.create_input_queue()


        # create the inference windows
        LOGGER.info("Creating the inference windows")
        self.create_windows()

        # load the first image
        self.next_image()

        # saveing variables
        self.out_idx = 0



    def create_windows(self):
        """
        Method that create all the windows:
            - One for the input image.
            - One for each class.
            - One for the output image.
        """

        self.input_image = None
        self.output_image = None

        cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)
        self.mask_windows = [
            MaskGuiWindow(name=class_name) for class_name in self.classes
        ]
        cv2.namedWindow("Output image", cv2.WINDOW_NORMAL)
    

    def create_input_queue(self):
        """
        Method that create the input queue.
        This method load all the names of all the images in the input folder
        """
        self.input_folder = self.config.input_images_folder
        self.input_idx = 0
        self.input_queue = [
            os.path.join(self.input_folder, file_name) for file_name in os.listdir(self.input_folder)
        ]
    

    def next_image(self):
        """
        Method that load the next image in the input queue.
        """
        self.input_idx += 1
        self.input_idx %= len(self.input_queue)
        self.input_image = cv2.imread(self.input_queue[self.input_idx])
        self.output_image = np.zeros_like(self.input_image)

        # show the new image
        cv2.imshow("Input image", self.input_image)
        cv2.imshow("Output image", self.output_image)

        # reset the masks
        MaskGuiWindow.reset(self.input_image.shape[:2])
    

    def inference(self):
        """
        Method that perform the inference.
        """
        self.output_image = self.model(
            self.input_image,
            MaskGuiWindow.get_masks()
        )

        # show the output image
        cv2.imshow("Output image", self.output_image)


    def save_sample(self):
        """
        Method that save the sample (input image, masks and output image)
        """
        # create the output folder
        output_folder = self.config.locations.samples_dir
        sample_folder = os.path.join(output_folder, f"sample_{self.out_idx}")
        os.makedirs(sample_folder, exist_ok=True)
        self.out_idx += 1

        # save the input image
        input_image_name = os.path.basename(self.input_queue[self.input_idx])
        input_image_path = os.path.join(sample_folder, f"input_{input_image_name}")
        cv2.imwrite(input_image_path, self.input_image)

        # save the masks
        for mask_window in self.mask_windows:
            mask_path = os.path.join(sample_folder, f"mask_{mask_window.name}.png")
            cv2.imwrite(mask_path, mask_window.mask)

        # save the output image
        output_image_path = os.path.join(sample_folder, f"output_{input_image_name}.jpg")
        cv2.imwrite(output_image_path, self.output_image)


    def help(self):
        """
        Method that print the help message.
        """
        print("""HELP: (key list)
        - Hold the right mouse button to draw the inpainting mask.
        - Hold the center mouse button to erase the inpainting mask.
        - Press 'h' to show this help message.
        - Press 'q' to quit the program.
        - Press <space bar> to load the next image.
        - Press 'w' to perform the inference.
        - Press 'r' to reset the masks.
        - Press 's' to save the sample (input image, masks and output image)
        - Press '+' to increase the brush size.
        - Press '-' to decrease the brush size.
        """)

    
    def run(self):
        """
        Method that run the inference GUI.
        This method start a loop that wait for the user inputs,
        and control the GUI workflow, trough the mouse and the keyboard.
        """
        self.help()

        while True:
            key = cv2.waitKey(0)

            if key == ord('q'):
                print("pressed 'q', quitting the program")
                break

            elif key == ord('h'):
                print("pressed 'h', showing the help message")
                self.help()

            elif key == ord(' '):
                print("pressed <space bar>, loading the next image")
                self.next_image()

            elif key == ord('w'):
                print("pressed 'w', performing the inference")
                self.inference()

            elif key == ord('r'):
                print("pressed 'r', resetting the masks")
                MaskGuiWindow.reset()
            
            elif key == ord('+'):
                print("pressed '+', increasing the brush size")
                MaskGuiWindow.set_brush_radius(MaskGuiWindow.brush_radius + 1)
                print(f"Brush radius set to: {MaskGuiWindow.brush_radius}")

            elif key == ord('-'):
                print("pressed '-', decreasing the brush size")
                if MaskGuiWindow.brush_radius > 1:
                    MaskGuiWindow.set_brush_radius(MaskGuiWindow.brush_radius - 1)
                    print(f"Brush radius set to: {MaskGuiWindow.brush_radius}")
                else:
                    print("Brush radius already at minimum value: 1")

            elif key == ord('s'):
                print("pressed 's', saving the sample")
                self.save_sample()
        
        cv2.destroyAllWindows()