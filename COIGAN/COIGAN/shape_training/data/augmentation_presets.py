from torchvision import transforms

augmentation_presets = {
    "preset_1": [
            transforms.RandomApply(
                [
                    transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), fillcolor=0),
                ],
                p=0.7
            )
        ],
    "preset_2": [
        #TODO add here something
    ]
}