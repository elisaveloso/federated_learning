import tensorflow as tf
from tensorflow.keras import layers, models, applications
from transformers import TFViTForImageClassification

def get_efficientnet_model(input_shape=(224, 224, 3), num_classes=1):
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = True # Fine-tuning

    inputs = layers.Input(shape=input_shape)
    # EfficientNet has built-in preprocessing (rescaling)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name="EfficientNetB0")
    return model

def get_resnet_model(input_shape=(224, 224, 3), num_classes=1):
    base_model = applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = True

    inputs = layers.Input(shape=input_shape)
    # ResNet50 expects specific preprocessing
    x = applications.resnet50.preprocess_input(inputs) 
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="ResNet50")
    return model

def get_vit_model(input_shape=(224, 224, 3), num_classes=1):
    # Load pre-trained ViT from Hugging Face
    # Using 'google/vit-base-patch16-224'
    model_id = 'google/vit-base-patch16-224'
    
    # Initialize the specific model
    vit_model = TFViTForImageClassification.from_pretrained(
        model_id,
        num_labels=1, # Binary classification for us? Or 2? 
        ignore_mismatched_sizes=True,
        use_safetensors=False
    )
    
  
    
    #inputs = layers.Input(shape=input_shape)
    inputs =  tf.keras.Input(shape=input_shape)
    
    # Hugging Face ViT ImageProcessor (default) usually does:
    # Resize to (224, 224), Rescale 1/255, Normalize (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
    # 1. Resize/Cast
    # x = layers.Resizing(224, 224)(inputs)
    # x = layers.Rescaling(1./127.5, offset=-1)(x) # Map 0..255 to -1..1 (approx for 0.5 mean/std)
    x = tf.transpose(inputs, perm=[0, 3, 1, 2])
     
    # 2. Transpose to channels_first if required by the *config*? 
   
    outputs = vit_model.vit(pixel_values=x)[0] 

    # Use the CLS token (index 0)
    cls_token = outputs[:, 0, :] 
    x = layers.Dropout(0.1)(cls_token)
    final_output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, final_output, name="ViT")
    return model

