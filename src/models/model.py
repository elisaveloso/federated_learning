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
        ignore_mismatched_sizes=True
    )
    
    # We need to wrap it in a Keras Functional API model if we want to add Resizing layers easily 
    # OR we just use it directly but data must be preprocessed.
    # To keep it consistent, let's wrap it and ensure input preprocessing.
    # ViT expects (batch, 3, 224, 224) if NCHW or (batch, 224, 224, 3) provided TF handles it?
    # TFViTForImageClassification main input is 'pixel_values'.
    # Hugging Face transformers usually expect preprocessed inputs. 
    # But for simplicity, let's build a Keras wrapper that includes resizing/normalization.
    
    inputs = layers.Input(shape=input_shape)
    
    # Hugging Face ViT ImageProcessor (default) usually does:
    # Resize to (224, 224), Rescale 1/255, Normalize (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    
    # 1. Resize/Cast
    x = layers.Resizing(224, 224)(inputs)
    x = layers.Rescaling(1./127.5, offset=-1)(x) # Map 0..255 to -1..1 (approx for 0.5 mean/std)
    
    # 2. Transpose to channels_first if required by the *config*? 
    # TFViT models typically accept channels_last (NHWC) in typical Keras flow
    # but let's check. Default TF implementation usually creates a model that accepts dict or tensor.
    # Let's rely on the internal Keras compatibility.
    
    # However, transformers models output an object (TFSequenceClassifierOutput).
    # We need the logits.
    
    # Limitation: Wrapping TFViTForImageClassification directly in Functional API 
    # can be tricky with saving/loading. 
    # An alternative is to just use a custom loop or sub-class.
    # Better yet: extract the 'vit' (encoder) part and add our head.
    
    # Simplified approach: Use the model as a layer.
    outputs = vit_model.vit(x)[0] # access the last hidden state
    # Use the CLS token (index 0)
    cls_token = outputs[:, 0, :] 
    x = layers.Dropout(0.1)(cls_token)
    final_output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, final_output, name="ViT")
    return model

