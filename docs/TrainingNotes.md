# Training Tests

## Training Test 1 (expirements/run_01_test)
Testing on the MNIST dataset. Using the following parameters (niter is the amnt of epochs). Training time set to 6hrs:

python dast.py --dataset=mnist --niter=10 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model

### Results
System fully functions and can train a substitute model. Test 1 was mainly for setup. 

Best model was netD_epoch_8.pth
Attack success rate: 6.95 %
Accuracy of the network on netD: 46.45 %

## Training Test 2 (expirements/run_02_test)
Testing on MNIST dataset. Using the following parameters. Increased Epoch size for test 2 to 80 epochs. Training time set to 32hrs:

python dast.py --dataset=mnist --niter=80 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model

### Results
Best model was netD_epoch_52.pth
Attack success rate: 20.75 %
Accuracy of the network on netD: 71.70 %

---

# Attack Tests
All attack tests are evaluated on model netD_epoch_52.pth from Training Test 2 (expirements/run_02_test/saved_models/netD_epoch_52.pth). This model produced the best Attack success rate and Accuracy of the network on netD at epoch 52 during training. Due to time constraints I was only able to train the model up to 80 epochs so for a reference a well trained MNIST model is around 99% accurate.

## FGSM

### Results (expirements/evaluation/FGSM/run_02_FGSM.txt)
Accuracy of the network on netD: 68.71 %
Attack success rate: 7.62 %
l2 distance:  4.1055 

### Notes
The substitue model classifies real MNIST images correctly 68.71% of the time despite being trained entirely on GAN-generated data showing that the generator was able to cover a reasonable portion of input space. Only 7.62% of the adversarial examples crafted from the substitute model fool the target model. This was kind of expected considering the accuracy gap. The average L2 perturbation distance of 4.1055 shows that meaningful perturbations were being used however they were not aligned well enough with the targets decision boundary to be effective.


