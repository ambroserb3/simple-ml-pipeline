# simple-ml-pipeline

## Usage

We use docker compose to build each stage in the pipeline, data-prep, train, eval, serve.

This builds images and creates containers, while mounting a data volume for the data-prep and train stage that the datasets and model tensors are stored in to be loaded during the train, evaluate, and deploy stages. 

1. `docker-compose up`
2. You can test the serve.py by exec-ing into the deploy container and running `curl -X POST -F "image=@tshirt-test.png" http://localhost:5000/predict`
   or you can test from this directory on the machine running the contain with `curl -X POST -F "image=@deploy/tshirt-test.png" http://localhost:5000/predict`

## Notes

I have created tests for the data prep and serving code, these are automatically run when the images are built.
