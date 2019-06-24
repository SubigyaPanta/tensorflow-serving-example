# How to use

Step 1: Run train_embed_save.py to train an example model.

Step 2: Run saved_to_served.py to convert the  *.hdf5 format model to 
servable model format.

Step 3: Run serve.sh (./serve.sh) to serve the servable model in a docker 
container as a web service.

Step 4: Run predict_from_serve.py to send a http request to the web 
service and recieve a json response of the prediction.

Step 5: Run tensorboard to visualize model