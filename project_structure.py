import os

def get_project_structure(base_path, exclude_folders=None, indent=0):
    """
    Recursively prints the project structure starting from base_path.
    
    :param base_path: The root directory of the project
    :param exclude_folders: List of folder names to exclude
    :param indent: Current indentation level for nested folders
    """
    exclude_folders = exclude_folders or []
    
    # Get the list of all files and folders in the current directory
    items = os.listdir(base_path)
    
    for item in sorted(items):
        # Full path of the item
        item_path = os.path.join(base_path, item)
        
        # Check if the item is a folder
        if os.path.isdir(item_path):
            if item in exclude_folders:
                continue
            print(" " * indent + f"├── {item}")
            get_project_structure(item_path, exclude_folders, indent + 4)
        else:
            print(" " * indent + f"├── {item}")

# Path to the root of your project
project_root = os.getcwd()  # Change this to your project's root directory if needed

# Call the function, excluding the 'models' folder
get_project_structure(project_root, exclude_folders=["models"])
