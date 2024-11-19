To run my code, you need to execute the main.ipynb notebook. it will download all my source code from githiub repository, and will 
also download the kaggle fingers dataset unprocessed. You can make sure that the data is unprocessed by checking in the data directory.
I had issues getting the code to work on colab or jupyter notebook due to needing to clone the git repo and dowload the data. However, from
my last check the code runs. In any case, I have also attached the source code in case that doe not work. That will run for sure locally, but 
ypu need to manually move the fingers data into the data directory.
The code expects the data to be like this:

data/
├── climate/
│   ├── test.csv   
│   ├── train.csv  
│         
├── fingers/
│   ├── train/
│   ├── test/
│        
├── train/
│    
├── test/
