 Libraries installation: New terminal
1. Python -m venv venv (First time only)
2. .\venv\Scripts\activate
3. pip install -r requiments.txt 
 
 
 How to setup your own YOUTUBE API KEY

 https://console.cloud.google.com/

 USING VENV IS RECOMMENDED

1. Create a project and select it
2. Menu -> APIs and Services -> Library
3. Search for youtube data api v3. Select and Enable it.
4. Create credentials.
5. Finish setting and get your own YOUTUBE API KEY
6. Create .env in your project and add it in .gitignore (Make sure you have python-dotenv)
7. YOUTUBE_API_KEY = "your-api-key-here"
