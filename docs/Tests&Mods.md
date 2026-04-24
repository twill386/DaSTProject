### Additions I added to the Code for testing

To view the synthetic images generated I added 2 lines to dast.py. Line 66-67 makes a directory to store the images. Line 454-455 Saves the synthetic images the generator produces after each epoch within a 10x10 grid. Since I am testing this using the MNIST dataset thats one image per digit class so 10 samples x 10 classes = 100 images.

### Test 1
