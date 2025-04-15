

#### Project Name : tiny-llm-colab-training

#### Purpose : To train simple LLM in colab-notebook and commit all codes to the GitHub for future changes and code versioning.

We will make a quick experiment using HuggingFace Transformers and datasets.

##### General configuration of GitHub repo in colab.

--clone your github repo on colab
"!git clone https://github.com/sumitrwk90/tiny-llm-colab-train.git"

--change to repo directory
"%cd tiny-llm-colab-train

--Befor you push the code you have to config your git user and email adderess
!git config --global user.email "sumitrwk90@gmail.com"
!git config --global user.name "sumitrwk90"

--Pass GitHub PassKey
from getpass import getpass
token = getpass('Enter your GitHub token: ')
!git remote set-url origin https://{token}@github.com/sumitrwk90/tiny-llm-colab-train.git

--Push your code to GitHub
!git status
!git add .
!git commit -m "______"
!git push origin main

--To pull files from GitHub
!git pull origin main

## 🤠 Author: SUMIT KUMAR

