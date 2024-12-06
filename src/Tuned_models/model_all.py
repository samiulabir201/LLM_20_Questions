from .model_gemma import delete_model as delete_gemma_model
from .model_deepmath import delete_model as delete_deepmath_model

def delete_all_models(exclude_model=None):
    """
    Deletes all loaded models except the one specified to exclude.
    
    Args:
        exclude_model (str, optional): The name of the model to exclude from deletion.
                                       Possible values: "gemma", "deepmath".
                                       Defaults to None, which deletes all models.
    """
    model_deletion_map = {
        "gemma": delete_gemma_model,
        "deepmath": delete_deepmath_model,
    }
    
    for model_name, delete_function in model_deletion_map.items():
        if model_name != exclude_model:
            delete_function()
