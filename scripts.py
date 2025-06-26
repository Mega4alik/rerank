def hf_push_to_hub():
    from huggingface_hub import HfApi, HfFolder, Repository, create_repo    
    repo_name = "AnuarSh/rerank1"    
    create_repo(repo_name, private=False)

    api = HfApi()
    api.upload_folder(
        folder_path="./model_temp/checkpoint-134000",
        repo_id=repo_name,
        repo_type="model",
    )

if __name__=="__main__":
	hf_push_to_hub()

