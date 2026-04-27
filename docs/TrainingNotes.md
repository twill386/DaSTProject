# Tests

## Test 1
Testing on the MNIST dataset. Using the following parameters (niter is the amnt of epochs). Training time set to 6hrs:

python dast.py --dataset=mnist --niter=10 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model

### Results

# Test 2
Testing on MNIST dataset. Using the following parameters. Increased Epoch size for test 2 to 80 epochs. Training time set to 32hrs:

python dast.py --dataset=mnist --niter=80 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model
