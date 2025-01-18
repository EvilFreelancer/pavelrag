import os
import shutil
from git import Repo

# Define the repository URL and the target subfolder
repo_url = "https://github.com/manticoresoftware/manticoresearch.git"
subfolder = "manual"
local_clone_path = "manticoresearch_clone"
local_manual_path = "manual_files"

# Clone the repository
print(f"Cloning repository from {repo_url}...")
Repo.clone_from(repo_url, local_clone_path)

# Define the path to the "manual" subfolder
manual_folder_path = os.path.join(local_clone_path, subfolder)

# Check if the "manual" subfolder exists
if not os.path.exists(manual_folder_path):
    print(f"The subfolder '{subfolder}' does not exist in the repository.")
    shutil.rmtree(local_clone_path)
    exit(1)

# Create a local directory to store the manual files
if os.path.exists(local_manual_path):
    print(f"Removing existing directory '{local_manual_path}'...")
    shutil.rmtree(local_manual_path)

# Copy the "manual" subfolder recursively to the local directory
print(f"Copying files and folders from '{subfolder}' to '{local_manual_path}'...")
shutil.copytree(manual_folder_path, local_manual_path)

# Clean up by removing the cloned repository
print("Cleaning up...")
shutil.rmtree(local_clone_path)

print(f"All files and folders from '{subfolder}' have been downloaded to '{local_manual_path}'.")
