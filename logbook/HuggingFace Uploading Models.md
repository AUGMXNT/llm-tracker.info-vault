https://huggingface.co/docs/hub/models-uploading

```
from huggingface_hub import create_repo, HfApi
create_repo("augmxnt/shisa-gamma-7b-v1")
api = HfApi()
api.upload_folder(
    folder_path="shisa-gamma-7b-v1",
    repo_id="augmxnt/shisa-gamma-7b-v1",
)
```