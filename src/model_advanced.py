import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("segmentation_models_pytorch not installed. Please install it via 'pip install segmentation-models-pytorch'")
    smp = None

def get_model(model_name, n_classes=3, encoder_name="resnet34", encoder_weights="imagenet"):
    """
    Factory function to get advanced segmentation models.
    
    Args:
        model_name (str): 'unet', 'unetplusplus', 'deeplabv3plus', 'manet'
        n_classes (int): Number of output classes
        encoder_name (str): Backbone encoder (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights (str): Pretrained weights ('imagenet' or None)
        
    Returns:
        torch.nn.Module: The segmentation model
    """
    if smp is None:
        raise ImportError("Please install segmentation-models-pytorch to use advanced models.")

    print(f"Creating {model_name} with encoder {encoder_name}...")
    
    # Logic to handle encoder_weights more robustly
    # If weights are 'imagenet' or 'noisy-student', we try to use them.
    # If connection fails, we fallback to None (random init) and warn the user.
    weights_to_use = encoder_weights
    
    # Helper to create model safely
    def create_model_safely(model_class, **kwargs):
        try:
            return model_class(**kwargs)
        except Exception as e:
            if kwargs.get('encoder_weights') is not None:
                print(f"Warning: Failed to load pretrained weights '{kwargs['encoder_weights']}' due to: {e}")
                print("Falling back to random initialization (encoder_weights=None).")
                kwargs['encoder_weights'] = None
                return model_class(**kwargs)
            else:
                raise e

    if model_name.lower() == 'unet':
        model = create_model_safely(
            smp.Unet,
            encoder_name=encoder_name, 
            encoder_weights=weights_to_use, 
            in_channels=3, 
            classes=n_classes,
        )
    elif model_name.lower() == 'unetplusplus':
        model = create_model_safely(
            smp.UnetPlusPlus,
            encoder_name=encoder_name, 
            encoder_weights=weights_to_use,
            in_channels=3, 
            classes=n_classes,
        )
    elif model_name.lower() == 'deeplabv3plus':
        model = create_model_safely(
            smp.DeepLabV3Plus,
            encoder_name=encoder_name, 
            encoder_weights=weights_to_use,
            in_channels=3, 
            classes=n_classes,
        )
    elif model_name.lower() == 'manet':
        model = create_model_safely(
            smp.MAnet,
            encoder_name=encoder_name, 
            encoder_weights=weights_to_use,
            in_channels=3, 
            classes=n_classes,
        )
    elif model_name.lower() == 'linknet':
        model = create_model_safely(
            smp.Linknet,
            encoder_name=encoder_name, 
            encoder_weights=weights_to_use,
            in_channels=3, 
            classes=n_classes,
        )
    elif model_name.lower() == 'pspnet':
        model = create_model_safely(
            smp.PSPNet,
            encoder_name=encoder_name, 
            encoder_weights=weights_to_use,
            in_channels=3, 
            classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

if __name__ == "__main__":
    # Test
    try:
        model = get_model("deeplabv3plus", n_classes=3)
        print("DeepLabV3+ created successfully")
        x = torch.randn(1, 3, 352, 480)
        y = model(x)
        print(f"Output shape: {y.shape}")
    except Exception as e:
        print(f"Error: {e}")
