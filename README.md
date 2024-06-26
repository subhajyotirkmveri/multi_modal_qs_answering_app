1. Clone this repository:
   
 ```
 git clone https://github.com/subhajyotirkmveri/multi_modal_qs_answering_app.git
 ```
2. Go to the cloning folder
```
cd multi_modal_qs_answering_app
```
3. Install all the depenedencies :
   
```
pip install -r requirements.txt
```
4. Need to dwonload the llama2 model or mistral model from the below link
```
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
```
or 
```
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
```
or 

```
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main
```
5. create a folder named 'models' at the same level as the Python files and  put the download model into 'models' folder 
6. Open terminal and run the following command:
```
streamlit run app_2.py
```
## Application Preview :
- Save the configuration file.
- Upload a PDF from the source.
- Click the process button to ingest the data.
- You can continue to ask questions.
![image](https://github.com/subhajyotirkmveri/multi_modal_qs_answering_app/blob/main/asset/asset_1.jpeg)
![image](https://github.com/subhajyotirkmveri/multi_modal_qs_answering_app/blob/main/asset/asset_2.jpeg)
