### Additions I added to the Code for testing

To view the synthetic images generated I added 2 lines to dast.py. Line 66-67 makes a directory to store the images. Line 454-455 Saves the synthetic images the generator produces after each epoch within a 10x10 grid. Since I am testing this using the MNIST dataset thats one image per digit class so 10 samples x 10 classes = 100 images.

# Test 1
Testing on the MNIST dataset. Using the following parameters (niter is the amnt of epochs):

python dast.py --dataset=mnist --niter=10 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model

### Error 1
When running the first test an error for the function zero_gradients occured. This was removed from Pytorch in version 2.x. advertorch is an older library that still tries to import it. This was fixed by injecting the function into torch prior to advertorch loading. 