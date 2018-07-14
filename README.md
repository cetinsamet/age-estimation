# age-estimation
Age estimation with PyTorch  
Age estimation is performed training a fully-connected network with 5000 sample face images.

## Usage  
$**python3**  age_est.py  input-image-path  
$**python3**  -W  ignore  age_est.py  input-image-path  *(**-W ignore** can be used to avoid warning messages)*  
  
### Example  
$**cd**  src  
$**python3**  -W  ignore  age_est.py  ../test.jpg  
Estimated Age : 27.7
  
![Example Image](https://github.com/cetinsamet/age-estimation/blob/master/test.jpg)  

*A photo taken after France won the 1998 FIFA World Cup.*  
*26-year-old Zidane is posing with the Cup.*  
*Age estimation of the model is 27.7*
