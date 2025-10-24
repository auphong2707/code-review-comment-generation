from huggingface_hub import HfApi, login
import os
import argparse

def authenticate_with_huggingface(token=None):
    """
    Authenticate with Hugging Face using a token.
    If token is not provided, it will look for it in the HUGGINGFACE_TOKEN environment variable.
    """
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if token is None:
            raise ValueError("No token provided and HUGGINGFACE_TOKEN environment variable is not set")
    
    login(token=token)
    return HfApi()

def upload_folder_to_huggingface(api, 
                                 folder_path, 
                                 repo_id, repo_type="model", 
                                 commit_message="Upload folder",
                                 allow_patterns="*official*/**",
                                 private=False):
    """
    Upload a folder to Hugging Face Hub.
    
    Args:
        api: HfApi instance
        folder_path: Path to the folder to upload
        repo_id: Repository ID in the format "username/repo_name"
        repo_type: Type of repository (default: "dataset", other options: "model", "space")
        commit_message: Commit message
        private: Whether the repository should be private
    """
    # Create the repository if it doesn't exist
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True
    )
    
    # Upload the folder
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        allow_patterns=allow_patterns
    )
    
    print(f"Successfully uploaded {folder_path} to {repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub")
    parser.add_argument("folder_path", help="Path to the folder to upload")
    parser.add_argument("repo_id", help="Repository ID in the format 'username/repo_name'")
    parser.add_argument("--allow_patterns", default="*official*/**", help="Pattern of file to upload to the folder")
    parser.add_argument("--token", help="Hugging Face token (if not provided, will look for HUGGINGFACE_TOKEN environment variable)")
    parser.add_argument("--repo-type", default="model", choices=["dataset", "model", "space"], help="Type of repository")
    parser.add_argument("--commit-message", default="Upload folder", help="Commit message")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    
    args = parser.parse_args()
    
    api = authenticate_with_huggingface(args.token)
    upload_folder_to_huggingface(
        api=api,
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        private=args.private,
        allow_patterns=args.allow_patterns
    )

if __name__ == "__main__":
    main()