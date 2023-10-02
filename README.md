![image](https://github.com/vitowen9580/H5-to-tensor-convertor/assets/53991729/a85069a2-db84-4970-a338-695e9d5e2ded)## 0. Enviroment
* Window10
* Python3.6
```bash
pip install --upgrade -r requirements.txt
````
## 1. Usage
* step1. Unet-based models training and put the *.h5 in folder "\resources"
(reference https://github.com/DebeshJha/ResUNetPlusPlus)
* Step2. Convert *.h5 into *.pb
  * Revised model_name in convert_h5_to_tensor.py 
  * run the python script
    ```bash
    python convert_h5_to_tensor.py
    ````
  * All .pb files are saved as "\application\pb_model" 
* Step3. *.h5 and *.pb performance comparison
* Step4. Convert *.pb into *.xml and *.bin by Openvino tool
  * cd C:\Program Files (x86)\Intel\openvino_2021.4.752\bin\
  ```bash
  setupvars.bat
  ````
  * Put the *.pb into "C:\Program Files (x86)\Intel\openvino_2021.4.752\deployment_tools\model_optimizer"
  ```bash
  python mo_tf.py --input_model ResUnet++_.pb --input_shape [1,256,256,3] --output_dir ./output/FP32  --data_type FP32
  ````
* Step 5. Inference
  Reference:

