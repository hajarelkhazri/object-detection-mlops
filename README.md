# Gérer des modèles sur le long terme

## Préparation de données

In this example , we implement an object detection ViT and we train it on the Caltech 101 dataset to detect an airplane in the given image.
You can download Caltech 101 datasets from this link: https://data.caltech.edu/records/mzrjq-6wc02
 

Here are the important steps:

-Import numpy,tensorflow,cv2,matplotlip.

-Prepare dataset:Change paths based in your file.

-Implement multilayer-perceptron (MLP).

-Implement the patch creation layer.

-Display patches for an input image.

-Implement the patch encoding layer:The PatchEncoder layer linearly transforms a patch by projecting it into a vector of size projection_dim. It also adds a learnable position embedding to the projected vector.

-Build the ViT model:The ViT model has multiple Transformer blocks. The MultiHeadAttention layer is used for self-attention, applied to the sequence of image patches. The encoded patches (skip connection) and self-attention layer outputs are normalized and fed into a multilayer perceptron (MLP). The model outputs four dimensions representing the bounding box coordinates of an object.

-Run the experiment.

-Evaluate the model.

But pay attention to paths .You should change cache_dir and unpack_archive paths

## Entraînement et tracking de modèle
On peut donc lancer notre entrainement, où nous utiliserons MLflow pour tracer les métriques(notamment la métrique de performance de notre modèle ici le f1-score), les paramètres,la version de la donnée et les artifacts de notre modèle, et aussi le packager.
A la fin du job d'entrainement, on peut visualiser les expérimentations avec l'interface utilisateur de Mlflow.

