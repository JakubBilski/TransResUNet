# TransResUNet: Improving U-Net Architecture for Robust Lungs Segmentation in Chest X-rays

python main.py should train the model and generate test results
in ../Data/input/our_test. Then, plot_dice_loss_vs_aug_type.py
generates plots.

Please note that loading the model doesn't work, so run on test data
is performed only once, only after training (main.py: 516-524).

Directory layout (it is also defined in main.py:167-188):

somerootfolder
    TransResUNet
        README.md
        ...
    Data
        input
            our_test
                Contrast-CHNCXR_0001_0.png
                ...
            our_test_masks
                Contrast-CHNCXR_0001_0_mask.png
                ...
            pulmonary-chest-xray-abnormalities
                ChinaSet_AllFiles
                    ...
                Montgomery
                    ...
            segmentation
                test
                    ...
                train
                    ...
