1. Clone this repository:
   
 ```
 git clone https://github.com/subhajyotirkmveri/multi_modal_qs_answering_app.git
 ```
2. Install all the depenedencies :
   
```
pip install -r requirements.txt
```
3. Need to dwonload the llama2 model or mistral model from the below link
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
4. create a folder named 'models' at the same level as the Python files and  put the download model into 'models' folder 
5. Open terminal and run the following command:
```
streamlit run app_2.py
```
## Application Preview :
![image](https://github.com/subhajyotirkmveri/multi_modal_qs_answering_app/asset/asset.jpeg)