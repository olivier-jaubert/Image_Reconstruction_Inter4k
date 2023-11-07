Dataset Inter4K Download
======================================================================================


Download via python (not always working):
=========================================
Run from project folder following python code:
    ```
    import os
    import gdown
    # DESTINATION FILE ON YOUR DISK
    folder = 'DatasetFolder'
    os.makedirs(folder,exist_ok=True)
    destination = os.path.join(folder,'Inter4K.zip')
    url = "https://drive.google.com/u/0/uc?id=1YNdzhk0mxFln9_2MetHF1cJIsoeVCskh&export=download"
    gdown.download(url, destination, quiet=False)

    os.system("unzip " + destination + " -d "+folder)
    ```

If the following error message appears please:

```
# Access denied with the following error:

#  	Cannot retrieve the public link of the file. You may need to change
# 	the permission to 'Anyone with the link', or have had many accesses. 

# You may still be able to access the file from the browser:

# 	 https://drive.google.com/u/0/uc?id=1YNdzhk0mxFln9_2MetHF1cJIsoeVCskh&export=download 
```

1) follow the link or go to the official webpage https://github.com/alexandrosstergiou/Inter4K and download via browser 
2) unzip dataset in DatasetFolder



Acknowledgments
===============
Inter4K Dataset courtesy of:


