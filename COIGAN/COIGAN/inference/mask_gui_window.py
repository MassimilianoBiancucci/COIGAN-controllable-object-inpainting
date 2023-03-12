import cv2
import numpy as np


class MaskGuiWindow:

    """
    Object that manage an istance of cv2 window,
    implementing the mouse callback function to draw,
    erase and move the masks.
    Further implement the callback to reset the mask,
    and finnally to get the mask.
    This class allow to modify the base mask dimension dinamically,
    with set_shape method.
    """

    windows = [] # list of the windows names
    masks = [] # list of the masks
    shape = (256, 256) # the same shape for all the masks
    flags = [] # list of the flags

    brush_radius = 10 # radius of the brush

    def __init__(
            self,
            name: str
        ):
        """
        Init method of the MaskGuiWindow object.

        Args:
            name (str): name of the window
        """

        # save the name
        self.name = name
        self.windows.append(self.name)
        self.masks.append(np.zeros(self.shape, dtype=np.uint8))

        # flags[0] = 1 -> draw, (left click keep pressed)
        # flags[1] = 1 -> erase, (middle click keep pressed)
        # if both 1 -> draw
        self.flags.append([0, 0]) 

        self.idx = len(MaskGuiWindow.windows) - 1

        # create the window
        cv2.namedWindow(self.name)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        
        # set the callbacks
        cv2.setMouseCallback(self.name, MaskGuiWindow.gui_callbacks, self.idx)

        # display the masks
        self.show()


    @property
    def mask(self):
        """
        Property that return the mask.
        """
        return self.masks[self.idx]


    @classmethod
    def set_shape(cls, shape: tuple):
        """
        Method that set the shape of the mask.

        Args:
            shape (tuple): shape of the mask
        """
        assert len(shape) == 2, "The shape must be a tuple of 2 elements"
        assert shape[0] > 0 and shape[1] > 0, "The shape must be positive and not null"
        cls.shape = shape
        cls.masks = [
            np.zeros(cls.shape, dtype=np.uint8)
            for _ in range(len(cls.windows))
        ]

        cls.show()


    @classmethod
    def set_brush_radius(cls, radius: int):
        """
        Method that set the radius of the brush.

        Args:
            radius (int): radius of the brush
        """
        assert radius > 0, "The radius must be positive and not null"
        cls.brush_radius = radius
        

    @classmethod
    def get_masks(cls, normalized: bool = True):
        """
        Method that return the masks in a single tensor of shape
        (height, width, n_classes)

        Args:
            normalized (bool, optional): if True, the masks are normalized between 0 and 1.
        
        Returns:
            np.ndarray: tensor of shape (height, width, n_classes)
        """
        if normalized:
            return np.stack(cls.masks, axis=-1) / 255
        return np.stack(cls.masks, axis=-1)


    @classmethod
    def show(cls):
        """
        Method that show all the windows.
        """
        for window, mask in zip(cls.windows, cls.masks):
            if mask is not None:
                cv2.imshow(window, mask)


    @classmethod
    def reset(cls, shape: tuple = None):
        """
        Method that reset all the masks.

        Args:
            shape (tuple, optional): shape of the mask. Defaults to None.
                    if not None, the shape of all the masks is set to this value.
        """
        if shape is not None:
            cls.set_shape(shape)
        else:
            for i in range(len(cls.windows)):
                cls.masks[i] = np.zeros(cls.shape, dtype=np.uint8)
            cls.show()


    @staticmethod
    def gui_callbacks(event, x, y, flags, params):
        """
        Callback function for the mouse ans keyboard events.
        """

        # get the window idx
        idx = params
        mask = MaskGuiWindow.masks[idx]
        flags = MaskGuiWindow.flags[idx]

        # if the mask is not created, return
        if MaskGuiWindow.masks[idx] is None:
            return

        # if the left button is pressed, draw mode
        if event == cv2.EVENT_LBUTTONDOWN: flags[0] = 1 # on
        if event == cv2.EVENT_LBUTTONUP: flags[0] = 0 # off

        # if the right button is pressed, erase mode
        if event == cv2.EVENT_MBUTTONDOWN: flags[1] = 1 # on
        if event == cv2.EVENT_MBUTTONUP: flags[1] = 0 # off

        # if the mouse is moved, draw
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags[0] == 1:
                cv2.circle(mask, (x, y), MaskGuiWindow.brush_radius, 255, -1)
            elif flags[1] == 1:
                cv2.circle(mask, (x, y), MaskGuiWindow.brush_radius, 0, -1)

        # update the window
        cv2.imshow(MaskGuiWindow.windows[idx], mask)


if __name__ == "__main__":

    # create 3 windows
    window1 = MaskGuiWindow('window1')
    window2 = MaskGuiWindow('window2')
    window3 = MaskGuiWindow('window3')

    # show the windows
    while True:
        if cv2.waitKey(0) == ord('q'):
            break
    
    cv2.destroyAllWindows()